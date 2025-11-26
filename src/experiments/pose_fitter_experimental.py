"""
experimental framework for testing mano ik improvements (cis 6800 final project).

this module extends pose_fitter.py with:
  1. configurable loss terms (ablation studies)
  2. alternative alignment methods (kabsch, weighted procrustes)
  3. joint limit constraints (anatomical feasibility)
  4. confidence-weighted losses (trust high-quality joints more)
  5. improved temporal smoothing (velocity + acceleration)
  6. comprehensive metrics collection (per-joint errors, convergence)

usage:
  # experiment 1: loss ablation
  config = ExperimentConfig(use_bone_loss=False)  # test without bone loss
  fitter = ExperimentalIKFitter(config)
  verts, joints, theta, metrics = fitter.fit(landmarks, mediapipe_confidence)

  # experiment 2: alignment methods
  config = ExperimentConfig(alignment_method="kabsch")  # rigid-only (no scale)
  fitter = ExperimentalIKFitter(config)

  # experiment 3: joint limits
  config = ExperimentConfig(use_joint_limits=True, lambda_limits=0.1)
  fitter = ExperimentalIKFitter(config)
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from model_utils.mano_model import CustomMANOLayer


# ============================================================
# experiment configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """configuration for ik optimization experiments."""

    # === loss term weights (for ablation studies) ===
    use_position_loss: bool = True
    lambda_position: float = 1.0

    use_bone_loss: bool = True
    lambda_bone: float = 0.5

    use_temporal_loss: bool = True
    lambda_temporal: float = 0.05

    use_regularization: bool = True
    lambda_reg: float = 1e-4

    use_joint_limits: bool = False  # NEW: anatomical constraints
    lambda_limits: float = 0.1

    use_confidence_weighting: bool = False  # NEW: weight by mediapipe confidence
    confidence_power: float = 2.0  # higher = more aggressive weighting

    use_velocity_smoothing: bool = False  # NEW: constrain velocity (not just position)
    lambda_velocity: float = 0.01

    # === alignment method ===
    alignment_method: str = "umeyama"  # options: "umeyama", "kabsch", "weighted_procrustes"
    use_scale: bool = True  # if False, rigid-only (no scale estimation)

    # === optimization parameters ===
    optimizer_type: str = "adam"  # options: "adam", "sgd", "lbfgs"
    learning_rate: float = 1e-2
    num_iterations: int = 15

    # === safety & fallback ===
    fallback_threshold: float = 0.5
    max_ik_error_mm: float = 50.0  # reject frames with error > this

    # === device ===
    device: str = "cpu"  # force cpu for svd compatibility


# ============================================================
# joint limit constraints (anatomical feasibility)
# ============================================================

class JointLimits:
    """
    anatomical joint angle limits for hand.

    mano pose parameters: θ ∈ R^45 = 15 joints × 3 DoF (axis-angle)
    limits specified in radians for each joint's rotation range.

    approximate physiological ranges (from biomechanics literature):
      - MCP (metacarpophalangeal): flexion/extension ~90°, abduction/adduction ~20°
      - PIP (proximal interphalangeal): flexion ~110°, no extension
      - DIP (distal interphalangeal): flexion ~80°, no extension
      - thumb: more complex, wider range of motion
    """

    def __init__(self):
        # store limits as [15 joints, 3 dof, 2 (min/max)]
        # axis-angle representation: magnitude = rotation angle, direction = axis
        self.limits = self._init_limits()

    def _init_limits(self) -> torch.Tensor:
        """
        initialize joint angle limits (conservative estimates).

        returns: [15, 3, 2] tensor with (min, max) for each joint's 3 DoF
        """
        # default: allow ±90° for all joints (conservative)
        limits = torch.ones(15, 3, 2) * torch.tensor([-np.pi/2, np.pi/2])

        # finger joints: restrict extension (can't bend backwards)
        # indices 1-14 are finger joints (excluding wrist at 0)
        for i in range(1, 15):
            limits[i, 0, 0] = 0  # no negative flexion (extension)
            limits[i, 0, 1] = np.pi * 0.6  # ~108° max flexion

        # thumb (indices 13-15): allow more abduction
        limits[13:16, 1, :] = torch.tensor([-np.pi/3, np.pi/3])  # ±60° abduction

        return limits

    def compute_violation_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """
        compute penalty for violating joint limits.

        uses smooth penalty: loss = sum(relu(theta - max)^2 + relu(min - theta)^2)
        (quadratic outside limits, zero inside)

        args:
          theta: [1, 45] or [45] pose parameters

        returns:
          loss: scalar tensor (0 if all joints within limits)
        """
        # reshape to [15, 3]
        if theta.dim() == 2:
            theta = theta.squeeze(0)
        theta_reshaped = theta.reshape(15, 3)

        # convert axis-angle to rotation magnitudes (approximate)
        # for simplicity, use L2 norm per joint
        # (true limit checking would require rodrigues → euler, more expensive)
        violations = torch.zeros(1, device=theta.device)

        for i in range(15):
            for j in range(3):
                val = theta_reshaped[i, j]
                min_val, max_val = self.limits[i, j, 0], self.limits[i, j, 1]

                # quadratic penalty outside limits
                if val < min_val:
                    violations += (min_val - val) ** 2
                elif val > max_val:
                    violations += (val - max_val) ** 2

        return violations


# ============================================================
# alternative alignment methods
# ============================================================

def kabsch_alignment(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    rigid-only alignment (no scale): min_{R,t} ||R·src + t - dst||²

    same as umeyama but forces s=1 (no scale estimation).
    useful for testing if scale estimation helps or hurts.

    returns: (s=1.0, R, t)
    """
    # center both clouds
    mu_src = src.mean(dim=0, keepdim=True)
    mu_dst = dst.mean(dim=0, keepdim=True)
    X = src - mu_src
    Y = dst - mu_dst

    # cross-covariance
    C = X.t() @ Y / src.shape[0]

    # svd
    U, S, Vt = torch.linalg.svd(C)
    R = U @ Vt

    # fix reflection
    if torch.det(R) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] = -Vt_fixed[-1, :]
        R = U @ Vt_fixed

    # translation (no scale)
    t = mu_dst.squeeze(0) - (R @ mu_src.squeeze(0))

    s = torch.tensor(1.0, device=src.device, dtype=src.dtype)

    return s, R, t


def umeyama_alignment(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    similarity transform: min_{s,R,t} ||s·R·src + t - dst||²

    (same as pose_fitter._umeyama, included here for completeness)
    """
    mu_src = src.mean(dim=0, keepdim=True)
    mu_dst = dst.mean(dim=0, keepdim=True)
    X = src - mu_src
    Y = dst - mu_dst

    C = X.t() @ Y / src.shape[0]
    U, S, Vt = torch.linalg.svd(C)
    R = U @ Vt

    if torch.det(R) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] = -Vt_fixed[-1, :]
        R = U @ Vt_fixed

    if with_scale:
        var_src = (X ** 2).sum() / src.shape[0]
        s = (S.sum() / var_src).clamp(min=1e-8)
    else:
        s = torch.tensor(1.0, device=src.device, dtype=src.dtype)

    t = mu_dst.squeeze(0) - s * (R @ mu_src.squeeze(0))

    return s, R, t


def weighted_procrustes(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor,
    with_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    confidence-weighted alignment: min_{s,R,t} sum_i w_i ||s·R·src_i + t - dst_i||²

    weights higher-confidence joints more in alignment estimation.
    useful when mediapipe confidence varies across joints.

    args:
      src: [N, 3] source points
      dst: [N, 3] target points
      weights: [N] confidence weights (0-1, sum to N)
      with_scale: estimate scale if True

    returns: (s, R, t)
    """
    # normalize weights to sum to N (for consistent scaling)
    W = weights / weights.sum() * src.shape[0]
    W = W.unsqueeze(1)  # [N, 1]

    # weighted centroids
    mu_src = (src * W).sum(dim=0, keepdim=True) / W.sum()
    mu_dst = (dst * W).sum(dim=0, keepdim=True) / W.sum()
    X = src - mu_src
    Y = dst - mu_dst

    # weighted cross-covariance
    C = (X.t() * W.t()) @ Y / W.sum()

    # svd
    U, S, Vt = torch.linalg.svd(C)
    R = U @ Vt

    if torch.det(R) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] = -Vt_fixed[-1, :]
        R = U @ Vt_fixed

    if with_scale:
        var_src = ((X ** 2) * W).sum() / W.sum()
        s = (S.sum() / var_src).clamp(min=1e-8)
    else:
        s = torch.tensor(1.0, device=src.device, dtype=src.dtype)

    t = mu_dst.squeeze(0) - s * (R @ mu_src.squeeze(0))

    return s, R, t


def apply_transform(pts: torch.Tensor, s: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """apply similarity transform: pts_aligned = s·R·pts + t"""
    return s * (pts @ R.t()) + t


# ============================================================
# experimental ik fitter
# ============================================================

class ExperimentalIKFitter:
    """
    inverse kinematics fitter with experimental features.

    supports:
      - configurable loss terms (ablation studies)
      - alternative alignment methods
      - joint limit constraints
      - confidence weighting
      - improved temporal smoothing
      - comprehensive metrics
    """

    def __init__(self, config: ExperimentConfig = None):
        """
        initialize fitter with experiment configuration.

        args:
          config: experiment configuration (uses defaults if None)
        """
        self.config = config or ExperimentConfig()
        self.device = torch.device(self.config.device)

        # load mano model
        mano_root = Path(__file__).resolve().parents[2] / "models"
        self.mano = CustomMANOLayer(
            mano_root=str(mano_root),
            side="right",
            device=str(self.device)
        )

        # joint mapping (mediapipe → mano)
        self.mp2mano = torch.tensor([
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
            10, 11, 12, 19, 7, 8, 9, 20
        ], device=self.device, dtype=torch.long)

        # bone structure (for bone direction loss)
        self.bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        # joint limits (if enabled)
        if self.config.use_joint_limits:
            self.joint_limits = JointLimits()

        # state (warm-start tracking)
        self.pose = torch.zeros(1, 45, device=self.device, dtype=torch.float32)
        self.betas = torch.zeros(1, 10, device=self.device, dtype=torch.float32)
        self.last_velocity = torch.zeros(1, 45, device=self.device, dtype=torch.float32)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.frame_count = 0

    def _compute_bone_dirs(self, joints: torch.Tensor) -> torch.Tensor:
        """compute normalized bone direction vectors."""
        directions = []
        for i, j in self.bones:
            vec = joints[j] - joints[i]
            norm = torch.norm(vec) + 1e-8
            directions.append(vec / norm)
        return torch.stack(directions, dim=0)

    def _get_alignment_fn(self):
        """get alignment function based on config."""
        method = self.config.alignment_method
        if method == "kabsch":
            return kabsch_alignment
        elif method == "umeyama":
            return umeyama_alignment
        elif method == "weighted_procrustes":
            return weighted_procrustes
        else:
            raise ValueError(f"Unknown alignment method: {method}")

    def fit(
        self,
        landmarks: np.ndarray,
        mediapipe_confidence: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        fit mano model to mediapipe landmarks.

        args:
          landmarks: [21, 3] mediapipe landmarks (meters)
          mediapipe_confidence: [21] confidence scores (0-1), optional
          verbose: print loss breakdown

        returns:
          vertices: [778, 3] aligned mano mesh
          joints: [21, 3] aligned mano joints
          theta: [45] optimized pose parameters
          metrics: dict with convergence metrics, per-joint errors, etc.
        """
        # convert inputs to torch
        mp_landmarks = torch.tensor(landmarks, device=self.device, dtype=torch.float32)

        if mediapipe_confidence is not None:
            mp_conf = torch.tensor(mediapipe_confidence, device=self.device, dtype=torch.float32)
        else:
            mp_conf = torch.ones(21, device=self.device, dtype=torch.float32)

        # initialize pose (warm-start from previous frame)
        pose = self.pose.clone().detach().requires_grad_(True)
        betas = self.betas.clone().detach()
        betas.requires_grad_(False)

        # setup optimizer
        if self.optimizer is None or self.config.optimizer_type == "lbfgs":
            # lbfgs needs fresh optimizer each frame
            if self.config.optimizer_type == "adam":
                self.optimizer = torch.optim.Adam([pose], lr=self.config.learning_rate)
            elif self.config.optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD([pose], lr=self.config.learning_rate)
            elif self.config.optimizer_type == "lbfgs":
                self.optimizer = torch.optim.LBFGS([pose], lr=self.config.learning_rate, max_iter=self.config.num_iterations)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
        else:
            self.optimizer.param_groups[0]['params'] = [pose]
            self.optimizer.param_groups[0]['lr'] = self.config.learning_rate

        # save previous state for temporal losses
        last_pose = self.pose.detach()
        last_velocity = self.last_velocity.detach()

        # get alignment function
        align_fn = self._get_alignment_fn()

        # metrics collection
        loss_history = []

        # optimization loop
        def closure():
            """closure for lbfgs optimizer (also used for adam/sgd)."""
            self.optimizer.zero_grad(set_to_none=True)

            # forward pass
            verts, joints = self.mano(pose, betas)
            joints = joints[0]  # [21, 3]
            mano_sel = joints[self.mp2mano]  # select corresponding joints

            # alignment
            if self.config.alignment_method == "weighted_procrustes":
                s, R, t = align_fn(mano_sel, mp_landmarks, mp_conf, with_scale=self.config.use_scale)
            else:
                s, R, t = align_fn(mano_sel, mp_landmarks, with_scale=self.config.use_scale)

            mano_aligned = apply_transform(mano_sel, s, R, t)

            # loss components
            total_loss = torch.tensor(0.0, device=self.device)
            loss_breakdown = {}

            # 1. position loss
            if self.config.use_position_loss:
                if self.config.use_confidence_weighting:
                    # weight by confidence^power
                    weights = mp_conf ** self.config.confidence_power
                    weights = weights / weights.mean()  # normalize
                    L_pos = (weights.unsqueeze(1) * (mano_aligned - mp_landmarks) ** 2).mean()
                else:
                    L_pos = ((mano_aligned - mp_landmarks) ** 2).mean()

                total_loss += self.config.lambda_position * L_pos
                loss_breakdown['position'] = L_pos.item()

            # 2. bone direction loss
            if self.config.use_bone_loss:
                mp_dirs = self._compute_bone_dirs(mp_landmarks)
                mano_dirs = self._compute_bone_dirs(mano_aligned)
                L_bone = ((mp_dirs - mano_dirs) ** 2).mean()
                total_loss += self.config.lambda_bone * L_bone
                loss_breakdown['bone'] = L_bone.item()

            # 3. temporal smoothness
            if self.config.use_temporal_loss:
                L_temporal = ((pose - last_pose) ** 2).mean()
                total_loss += self.config.lambda_temporal * L_temporal
                loss_breakdown['temporal'] = L_temporal.item()

            # 4. regularization
            if self.config.use_regularization:
                L_reg = (pose ** 2).mean()
                total_loss += self.config.lambda_reg * L_reg
                loss_breakdown['reg'] = L_reg.item()

            # 5. joint limits (NEW)
            if self.config.use_joint_limits:
                L_limits = self.joint_limits.compute_violation_loss(pose)
                total_loss += self.config.lambda_limits * L_limits
                loss_breakdown['limits'] = L_limits.item()

            # 6. velocity smoothing (NEW)
            if self.config.use_velocity_smoothing:
                current_velocity = pose - last_pose
                L_velocity = ((current_velocity - last_velocity) ** 2).mean()
                total_loss += self.config.lambda_velocity * L_velocity
                loss_breakdown['velocity'] = L_velocity.item()

            loss_breakdown['total'] = total_loss.item()
            loss_history.append(loss_breakdown)

            total_loss.backward()
            return total_loss

        # run optimization
        if self.config.optimizer_type == "lbfgs":
            self.optimizer.step(closure)
        else:
            for step in range(self.config.num_iterations):
                closure()
                self.optimizer.step()

        # final forward pass
        with torch.no_grad():
            verts, joints = self.mano(pose, betas)
            verts = verts[0]
            joints = joints[0]
            mano_sel = joints[self.mp2mano]

            # final alignment
            if self.config.alignment_method == "weighted_procrustes":
                s, R, t = align_fn(mano_sel, mp_landmarks, mp_conf, with_scale=self.config.use_scale)
            else:
                s, R, t = align_fn(mano_sel, mp_landmarks, with_scale=self.config.use_scale)

            verts_aligned = apply_transform(verts, s, R, t)
            joints_aligned = apply_transform(joints, s, R, t)
            mano_aligned = apply_transform(mano_sel, s, R, t)

            # compute metrics
            joint_errors = torch.norm(mano_aligned - mp_landmarks, dim=1)  # [21]
            ik_error_mm = joint_errors.mean().item() * 1000

            metrics = {
                'ik_error_mm': ik_error_mm,
                'per_joint_error_mm': (joint_errors * 1000).cpu().numpy(),
                'loss_history': loss_history,
                'final_loss': loss_history[-1] if loss_history else {},
                'scale': s.item(),
                'converged': ik_error_mm < self.config.max_ik_error_mm,
            }

        # update state for next frame
        self.pose = pose.detach()
        self.last_velocity = (pose - last_pose).detach()
        self.frame_count += 1

        # convert to numpy
        verts_np = verts_aligned.detach().cpu().numpy().astype(np.float32)
        joints_np = joints_aligned.detach().cpu().numpy().astype(np.float32)
        theta_np = pose[0].detach().cpu().numpy().astype(np.float32)

        if verbose:
            print(f"Frame {self.frame_count}: IK error = {ik_error_mm:.2f} mm")
            print(f"  Loss breakdown: {metrics['final_loss']}")

        return verts_np, joints_np, theta_np, metrics


# ============================================================
# batch evaluation (for experiments)
# ============================================================

def run_experiment(
    video_or_data: str,
    config: ExperimentConfig,
    output_path: Optional[str] = None
) -> Dict:
    """
    run ik fitting on video/dataset with given config.

    collects comprehensive metrics for analysis:
      - per-frame ik errors
      - per-joint error distributions
      - convergence statistics
      - loss term contributions

    args:
      video_or_data: path to video file or recorded session
      config: experiment configuration
      output_path: save results to this path (optional)

    returns:
      results: dict with all metrics
    """
    # TODO: implement video loading + frame-by-frame fitting
    # for now, return placeholder
    results = {
        'config': config,
        'frames_processed': 0,
        'mean_ik_error_mm': 0.0,
        'per_joint_errors': [],
        'convergence_rate': 0.0,
    }

    return results


# ============================================================
# quick test
# ============================================================

if __name__ == "__main__":
    print("Experimental IK Fitter - CIS 6800 Final Project")
    print("=" * 60)

    # example: test different configurations
    configs = {
        "baseline": ExperimentConfig(),
        "no_bone_loss": ExperimentConfig(use_bone_loss=False),
        "no_temporal": ExperimentConfig(use_temporal_loss=False),
        "with_limits": ExperimentConfig(use_joint_limits=True, lambda_limits=0.1),
        "kabsch_only": ExperimentConfig(alignment_method="kabsch", use_scale=False),
    }

    print("\nExperiment configurations loaded:")
    for name, cfg in configs.items():
        print(f"  - {name}")

    print("\nReady to run experiments!")
    print("Usage:")
    print("  fitter = ExperimentalIKFitter(configs['baseline'])")
    print("  verts, joints, theta, metrics = fitter.fit(landmarks)")
