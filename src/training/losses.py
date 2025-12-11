"""
enhanced loss functions for mano hand pose training.

includes:
  - manoloss with balanced lambda_joint=5.0
  - quaternion geodesic angular loss for joint rotations
  - per-joint weighting (proximal 2.0, distal 1.0, higher for pinky/thumb)
  - kinematic priors: soft tanh clipping on θ ranges, gaussian pca prior
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


def axis_angle_to_quaternion(theta: torch.Tensor) -> torch.Tensor:
    """convert axis-angle representation to quaternion. returns [B, 4] quaternions (w, x, y, z)."""
    angle = torch.norm(theta, dim=1, keepdim=True)  # [B, 1]
    axis = theta / (angle + 1e-8)  # [B, 3]
    
    half_angle = angle * 0.5
    w = torch.cos(half_angle)  # [B, 1]
    xyz = axis * torch.sin(half_angle)  # [B, 3]
    
    quat = torch.cat([w, xyz], dim=1)  # [B, 4]
    return quat


def quaternion_geodesic_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    compute geodesic distance between quaternions (angular loss).
    
    geodesic distance: d(q1, q2) = arccos(|q1 · q2|)
    captures rotation differences that cartesian mpjpe misses (twists).
    returns [B] geodesic distances in radians.
    """
    # normalize quaternions
    q1_norm = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
    q2_norm = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)
    
    # dot product (quaternion inner product)
    dot = (q1_norm * q2_norm).sum(dim=1)  # [B]
    
    # clamp to [-1, 1] for arccos
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # geodesic distance (angular difference)
    geodesic = torch.acos(torch.abs(dot))  # [B]
    
    return geodesic


def get_per_joint_weights() -> torch.Tensor:
    """
    get per-joint weights for mano loss.
    
    based on exp4 analysis: proximal joints more reliable, fingertips higher error.
    weights: proximal 2.0, distal 1.0, higher for pinky/thumb (harder to track).
    
    mano joint order (15 joints × 3 dof = 45 params):
      0: wrist (global)
      1-3: thumb (mcp, pip, dip)
      4-6: index (mcp, pip, dip)
      7-9: middle (mcp, pip, dip)
      10-12: ring (mcp, pip, dip)
      13-15: pinky (mcp, pip, dip)
    
    returns [15] per-joint weights.
    """
    weights = torch.ones(15)
    
    # proximal joints (mcp): 2.0
    proximal_indices = [1, 4, 7, 10, 13]  # thumb/index/middle/ring/pinky mcp
    for idx in proximal_indices:
        weights[idx] = 2.0
    
    # distal joints (pip, dip): 1.0 (default)
    # already set to 1.0
    
    # pinky and thumb: higher weights (harder to track per exp4)
    pinky_indices = [13, 14, 15]  # pinky mcp, pip, dip
    thumb_indices = [1, 2, 3]  # thumb mcp, pip, dip
    for idx in pinky_indices + thumb_indices:
        weights[idx] *= 1.5  # 2.0 * 1.5 = 3.0 for mcp, 1.0 * 1.5 = 1.5 for pip/dip
    
    return weights


class MANOLoss(nn.Module):
    """
    enhanced multi-term loss for mano training.
    
    features:
      - balanced lambda_joint=5.0 (reduced from 20.0)
      - quaternion geodesic angular loss on joint rotations
      - per-joint weighting (proximal 2.0, distal 1.0, higher for pinky/thumb)
      - kinematic priors: soft tanh clipping on θ ranges, gaussian pca prior
    """
    
    def __init__(
        self,
        lambda_param: float = 0.5,
        lambda_joint: float = 5.0,  # reduced from 20.0 per synthesis
        lambda_angular: float = 1.0,  # quaternion geodesic loss
        lambda_reg: float = 0.001,
        lambda_pca: float = 0.1,  # pca prior
        use_vertex_loss: bool = False,
        mano_layer: Optional[object] = None,
        theta_range: Tuple[float, float] = (-np.pi/2, np.pi/2),  # soft clipping range
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_joint = lambda_joint
        self.lambda_angular = lambda_angular
        self.lambda_reg = lambda_reg
        self.lambda_pca = lambda_pca
        self.use_vertex_loss = use_vertex_loss
        self.mano_layer = mano_layer
        self.theta_range = theta_range
        
        self.mse_loss = nn.MSELoss(reduction='none')  # per-element for weighting
        self.joint_weights = get_per_joint_weights()  # [15] per-joint weights
        
        # load mano mean and pca components if available
        self.mano_mean = None
        self.mano_components = None
        if mano_layer is not None and hasattr(mano_layer, 'hands_mean'):
            self.mano_mean = mano_layer.hands_mean  # [45]
            if hasattr(mano_layer, 'hands_components') and mano_layer.hands_components is not None:
                self.mano_components = mano_layer.hands_components  # [N, 45] PCA components
    
    def soft_tanh_clip(self, theta: torch.Tensor) -> torch.Tensor:
        """
        soft clipping on θ ranges using tanh.
        
        instead of hard clamping, use tanh to smoothly constrain values.
        preserves gradients while encouraging anatomically plausible poses.
        returns [B, 45] softly clipped parameters.
        """
        # normalize to [-1, 1] range, then scale to theta_range
        theta_normalized = (theta - self.theta_range[0]) / (self.theta_range[1] - self.theta_range[0]) * 2.0 - 1.0
        theta_tanh = torch.tanh(theta_normalized)
        theta_clipped = (theta_tanh + 1.0) * 0.5 * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        return theta_clipped
    
    def pca_prior_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """
        gaussian pca prior on pose space.
        
        encourages poses to stay near mano's learned pose distribution.
        uses mano mean and pca components if available.
        returns [B] pca prior losses.
        """
        if self.mano_mean is None:
            # no pca available, return zero loss
            return torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        
        # project onto pca space (if components available)
        if self.mano_components is not None:
            # center around mean
            theta_centered = theta - self.mano_mean.unsqueeze(0)  # [B, 45]
            
            # project onto pca components
            # theta_pca = theta_centered @ self.mano_components.T  # [B, N]
            
            # use first few components (most variance)
            n_components = min(10, self.mano_components.shape[0])
            components = self.mano_components[:n_components]  # [N, 45]
            
            # project and compute reconstruction error
            theta_pca = torch.matmul(theta_centered, components.T)  # [B, N]
            theta_recon = torch.matmul(theta_pca, components)  # [B, 45]
            reconstruction_error = torch.norm(theta_centered - theta_recon, dim=1)  # [B]
            
            return reconstruction_error
        else:
            # simple gaussian prior: distance from mean
            theta_centered = theta - self.mano_mean.unsqueeze(0)  # [B, 45]
            return torch.norm(theta_centered, dim=1)  # [B]
    
    def forward(
        self,
        pred_theta: torch.Tensor,
        pred_beta: torch.Tensor,
        gt_theta: torch.Tensor,
        gt_beta: torch.Tensor,
        gt_joints: Optional[torch.Tensor] = None,
        gt_vertices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        compute enhanced mano loss.
        
        returns dict with 'param', 'joint', 'angular', 'reg', 'pca', 'total'.
        """
        losses = {}
        B = pred_theta.shape[0]
        
        # 1. parameter loss (l2)
        loss_param_theta = self.mse_loss(pred_theta, gt_theta).mean()
        loss_param_beta = self.mse_loss(pred_beta, gt_beta).mean()
        loss_param = loss_param_theta + 0.1 * loss_param_beta
        losses['param'] = loss_param
        
        # 2. quaternion geodesic angular loss (captures twists)
        # reshape theta: [B, 45] -> [B, 15, 3] (15 joints × 3 dof)
        pred_theta_reshaped = pred_theta.view(B, 15, 3)  # [B, 15, 3]
        gt_theta_reshaped = gt_theta.view(B, 15, 3)  # [B, 15, 3]
        
        # convert to quaternions
        pred_quat = axis_angle_to_quaternion(pred_theta_reshaped.view(-1, 3))  # [B*15, 4]
        gt_quat = axis_angle_to_quaternion(gt_theta_reshaped.view(-1, 3))  # [B*15, 4]
        
        # compute geodesic loss per joint
        angular_loss_per_joint = quaternion_geodesic_loss(pred_quat, gt_quat)  # [B*15]
        angular_loss_per_joint = angular_loss_per_joint.view(B, 15)  # [B, 15]
        
        # apply per-joint weights
        joint_weights = self.joint_weights.to(pred_theta.device)  # [15]
        angular_loss_weighted = (angular_loss_per_joint * joint_weights.unsqueeze(0)).mean()  # scalar
        losses['angular'] = angular_loss_weighted
        
        # 3. joint position loss (with per-joint weighting)
        if gt_joints is not None and self.mano_layer is not None:
            # soft clip mano parameters
            pred_theta_clipped = self.soft_tanh_clip(pred_theta)
            pred_beta_clipped = torch.clamp(pred_beta, -3.0, 3.0)
            
            # forward pass
            pred_verts, pred_joints = self.mano_layer(pred_theta_clipped, pred_beta_clipped)
            pred_joints = pred_joints.float()
            gt_joints = gt_joints.float()
            
            # check for nan
            if torch.isnan(pred_joints).any() or torch.isinf(pred_joints).any():
                loss_joint = pred_theta.sum() * 0.0 + 100.0
            else:
                # per-joint mse: [B, 21]
                joint_errors = self.mse_loss(pred_joints, gt_joints).mean(dim=2)  # [B, 21]
                
                # map 21 joints to 15 mano joints (wrist + 5 fingers × 3 joints each)
                # mano has 16 skeleton joints, mediapipe has 21 (includes fingertips)
                # for simplicity, use uniform weighting across all 21 joints
                # (in practice, you'd map mediapipe joints to mano joints)
                loss_joint = joint_errors.mean()
            losses['joint'] = loss_joint
        else:
            loss_joint = pred_theta.sum() * 0.0
            losses['joint'] = loss_joint
        
        # 4. vertex loss (optional)
        if self.use_vertex_loss and gt_vertices is not None and self.mano_layer is not None:
            if 'pred_verts' not in locals():
                pred_theta_clipped = self.soft_tanh_clip(pred_theta)
                pred_beta_clipped = torch.clamp(pred_beta, -3.0, 3.0)
                pred_verts, _ = self.mano_layer(pred_theta_clipped, pred_beta_clipped)
            pred_verts = pred_verts.float()
            gt_vertices = gt_vertices.float()
            loss_vertex = self.mse_loss(pred_verts, gt_vertices).mean()
            losses['vertex'] = loss_vertex
        else:
            loss_vertex = pred_theta.sum() * 0.0
            losses['vertex'] = loss_vertex
        
        # 5. regularization
        loss_reg_theta = (pred_theta ** 2).mean()
        loss_reg_beta = (pred_beta ** 2).mean()
        loss_reg = loss_reg_theta + loss_reg_beta
        losses['reg'] = loss_reg
        
        # 6. pca prior loss
        loss_pca = self.pca_prior_loss(pred_theta).mean()
        losses['pca'] = loss_pca
        
        # total loss
        total_loss = (
            self.lambda_param * loss_param +
            self.lambda_joint * loss_joint +
            self.lambda_angular * angular_loss_weighted +
            self.lambda_reg * loss_reg +
            self.lambda_pca * loss_pca
        )
        
        # check for nan/inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = pred_theta.sum() * 0.0 + 1000.0
        
        losses['total'] = total_loss
        
        return losses


class SimpleMANOLoss(nn.Module):
    """simplified loss: just parameter l2 + regularization. fast, no mano forward pass."""
    
    def __init__(self, lambda_param: float = 1.0, lambda_reg: float = 0.01):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_reg = lambda_reg
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        pred_theta: torch.Tensor,
        pred_beta: torch.Tensor,
        gt_theta: torch.Tensor,
        gt_beta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # parameter loss
        loss_param_theta = self.mse_loss(pred_theta, gt_theta)
        loss_param_beta = self.mse_loss(pred_beta, gt_beta)
        loss_param = loss_param_theta + 0.1 * loss_param_beta
        
        # regularization
        loss_reg = (pred_theta ** 2).mean() + (pred_beta ** 2).mean()
        
        # total loss
        total_loss = self.lambda_param * loss_param + self.lambda_reg * loss_reg
        
        return {
            'param': loss_param,
            'reg': loss_reg,
            'total': total_loss,
        }
