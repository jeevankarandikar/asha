"""
interface to mano parametric hand model with inverse kinematics fitting.

this module provides two ways to convert mediapipe landmarks to mano mesh:

1. mano_from_landmarks_simple(): neutral pose (fast, no optimization)
   - use for: testing, fallback, low-power devices
   - returns: neutral hand shape V_shaped (778 vertices)
   - performance: <0.5ms per frame

2. mano_from_landmarks(): full IK fitting (slower, accurate articulation)
   - solves: min_θ ||sRJ(θ) + t - X||² + λ_bone·bone_loss + λ_smooth·smooth_loss + λ_reg·reg_loss
   - uses: umeyama alignment, adam optimizer, warm-start tracking
   - ~1-4ms per frame on M1/M2 (25 optimization steps)
   - includes fallback to simple mode if IK diverges

math overview:
  X ∈ R^21×3 : mediapipe landmarks (input, preferably world coordinates in meters)
  θ ∈ R^45   : mano pose parameters (15 joints × 3 DoF axis-angle rotations)
  β ∈ R^10   : shape parameters (hand morphology: thin/thick, long/short fingers)
  (s,R,t)    : similarity transform (scale, rotation ∈ SO(3), translation)
  J(θ,β)     : mano joints (output of forward pass)

  we optimize θ to match J(θ) to X after rigid alignment (s,R,t) via procrustes/umeyama.

  loss terms:
    - joint position: ||sRJ_Π(θ) + t - X||²_F  (L2 after alignment)
    - bone direction: ||D(sRJ) - D(X)||²_F      (scale-invariant, D normalizes edge vectors)
    - temporal smooth: ||θ - θ_{t-1}||²         (stay close to previous frame)
    - regularization: ||θ||²                     (prefer small angles)

see function docstrings for detailed math explanations.

current limitations (V1):
  - mano forward pass returns V_shaped (neutral pose), not full LBS with joint rotations
  - IK optimizes θ but articulation comes mainly from rigid alignment, not pose blend shapes
  - future V2: wire full LBS (rodrigues → FK → posedirs → skinning) for proper finger bending
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
from .mano_model import CustomMANOLayer


# ============================================================
# IK configuration (tune these for your setup)
# ============================================================

# optimization parameters
IK_STEPS = 15                # number of adam iterations per frame
                            # more steps = tighter fit, slower
                            # fewer steps = faster, may lag on quick motions
                            # typical range: 10-30 (15 is good for CPU, 25 for MPS)
                            # note: reduced to 15 because CPU-only mode (MPS lacks SVD support)

LEARNING_RATE = 1e-2        # adam learning rate
                            # higher = faster convergence, risk of instability
                            # lower = slower, more stable
                            # typical range: 5e-3 to 2e-2

# loss function weights
WEIGHT_JOINT = 1.0          # λ for joint position loss: ||sRJ(θ)+t - X||²
                            # this is the primary fitting term
                            # keep at 1.0, adjust others relative to this

WEIGHT_BONE = 0.5           # λ for bone direction loss: ||D(sRJ) - D(X)||²
                            # scale-invariant, helps articulation when depth is noisy
                            # increase (0.7-1.0) if fingers bend incorrectly
                            # decrease (0.2-0.4) if tracking is too rigid

WEIGHT_SMOOTH = 0.05        # λ for temporal smoothness: ||θ - θ_{t-1}||²
                            # kills jitter, keeps pose close to previous frame
                            # increase (0.1-0.3) if tracking jitters
                            # decrease (0.01-0.03) if tracking lags behind motion

WEIGHT_REG = 1e-4           # λ for pose regularization: ||θ||²
                            # prefers small joint angles (neutral pose)
                            # helps when observations are noisy/ambiguous
                            # typical range: 1e-5 to 1e-3

# safety & fallback
FALLBACK_THRESHOLD = 0.5    # max acceptable loss before fallback to simple mode
                            # if final loss > this, IK likely diverged
                            # return neutral pose instead of bad fit
                            # increase if too many false fallbacks
                            # decrease if seeing impossible poses slip through


# ============================================================
# device and model setup
# ============================================================

def get_torch_device() -> torch.device:
    """
    pick best device: mps (apple silicon gpu) if available, otherwise cpu.

    note: currently forcing CPU because MPS doesn't support linalg.svd (used in umeyama).
    the fallback causes expensive device transfers (~100ms per frame).
    once pytorch adds MPS support for SVD, can switch back to MPS.
    """
    # temporarily use CPU to avoid MPS→CPU fallback overhead
    return torch.device("cpu")

    # original (enable once MPS supports SVD):
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    # return torch.device("cpu")


# where mano model files live (project root, not src/)
_MANO_ROOT = Path(__file__).resolve().parents[2] / "models" / "mano"
_DEVICE = get_torch_device()

# singleton - load model once and reuse
# none until first use (lazy loading)
_mano_layer: CustomMANOLayer | None = None


def _lazy_load_layer(side: str = "right") -> CustomMANOLayer:
    """load mano model once (singleton pattern). creates on first call, reuses after."""
    global _mano_layer
    if _mano_layer is None:
        print(f"loading mano model ({side} hand) on device: {_DEVICE}")
        _mano_layer = CustomMANOLayer(
            mano_root=str(_MANO_ROOT),
            side=side,
            device=str(_DEVICE)
        )
        print(f"mano model loaded successfully")
    return _mano_layer


def get_mano_faces() -> np.ndarray:
    """get triangle faces for mesh. returns which vertices form each triangle (needed for rendering)."""
    layer = _lazy_load_layer()
    # move from gpu to cpu memory, convert to numpy
    return layer.faces.detach().cpu().numpy()


# ============================================================
# kinematic structures (mediapipe ↔ MANO correspondence)
# ============================================================

# mediapipe hand connections (parent→child bone pairs)
# defines the kinematic tree topology for bone direction constraints
# 20 bones total: 4 per finger (thumb through pinky)
_MP_BONES = [
    # thumb: CMC → MCP → IP → tip
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index: MCP → PIP → DIP → tip
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle: MCP → PIP → DIP → tip
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring: MCP → PIP → DIP → tip
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky: MCP → PIP → DIP → tip
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# joint mapping: mediapipe index → MANO index (permutation matrix Π)
# both have 21 joints, but ordering differs:
#   mediapipe: wrist(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
#   MANO: first 16 from J_regressor, then 5 fingertip vertices appended
#
# IMPORTANT: this mapping depends on your specific MANO model build
# the J_regressor can order the 16 skeleton joints differently
# use debug_print_joint_order() to validate and refine this mapping
#
# how to validate:
#   1. run debug_print_joint_order() once with neutral pose
#   2. visually inspect joint positions (wrist at origin, fingers extending)
#   3. adjust indices below so chains match anatomically:
#      - wrist should map to wrist
#      - finger bases should map to MCPs
#      - fingertips should map to tip vertices (indices 16-20 in MANO output)
_MANO21_FOR_MP = np.array([
    0,   # mp 0: wrist  → mano 0: wrist (root joint)
    13,  # mp 1: thumb_CMC → mano 13: thumb base
    14,  # mp 2: thumb_MCP → mano 14: thumb MCP
    15,  # mp 3: thumb_IP  → mano 15: thumb IP
    16,  # mp 4: thumb_tip → mano 16: thumb fingertip vertex
    1,   # mp 5: index_MCP → mano 1: index base
    2,   # mp 6: index_PIP → mano 2: index PIP
    3,   # mp 7: index_DIP → mano 3: index DIP
    17,  # mp 8: index_tip → mano 17: index fingertip vertex
    4,   # mp 9: middle_MCP → mano 4: middle base
    5,   # mp 10: middle_PIP → mano 5: middle PIP
    6,   # mp 11: middle_DIP → mano 6: middle DIP
    18,  # mp 12: middle_tip → mano 18: middle fingertip vertex
    10,  # mp 13: ring_MCP → mano 10: ring base (note: may need swap with pinky)
    11,  # mp 14: ring_PIP → mano 11: ring PIP
    12,  # mp 15: ring_DIP → mano 12: ring DIP
    19,  # mp 16: ring_tip → mano 19: ring fingertip vertex
    7,   # mp 17: pinky_MCP → mano 7: pinky base
    8,   # mp 18: pinky_PIP → mano 8: pinky PIP
    9,   # mp 19: pinky_DIP → mano 9: pinky DIP
    20,  # mp 20: pinky_tip → mano 20: pinky fingertip vertex
], dtype=np.int64)


# ============================================================
# umeyama alignment (closed-form procrustes/similarity transform)
# ============================================================

def _umeyama(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    solve for optimal similarity transform: min_{s,R,t} ||s·R·src + t·1ᵀ - dst||²_F
    returns (scale s, rotation R ∈ SO(3), translation t) via SVD (umeyama/procrustes).

    algorithm (horn's method / kabsch-umeyama):
      1. center both point clouds: X = src - μ_src, Y = dst - μ_dst
      2. cross-covariance: C = XᵀY / n
      3. SVD: C = UΣVᵀ
      4. rotation: R = UVᵀ (with det check for proper rotation)
      5. scale: s = tr(Σ) / var(X)  [if with_scale=True]
      6. translation: t = μ_dst - s·R·μ_src

    args:
      src: [N, 3] source points (e.g., MANO joints)
      dst: [N, 3] target points (e.g., mediapipe landmarks)
      with_scale: if True, solve for scale; else assume s=1 (rigid only)

    returns:
      s: scalar scale (tensor)
      R: [3,3] rotation matrix (proper, det(R)=+1)
      t: [3] translation vector

    why this works:
      - removes global pose ambiguity (MANO model space ≠ camera space)
      - closed-form solution (no optimization needed)
      - recomputed every iteration as θ changes J(θ)
      - makes IK focus on articulation, not global alignment
    """
    assert src.shape == dst.shape, f"src {src.shape} and dst {dst.shape} must match"

    # step 1: center both point clouds (remove translation)
    mu_src = src.mean(dim=0, keepdim=True)  # [1, 3]
    mu_dst = dst.mean(dim=0, keepdim=True)  # [1, 3]
    X = src - mu_src  # [N, 3]
    Y = dst - mu_dst  # [N, 3]

    # step 2: cross-covariance matrix
    C = X.t() @ Y / src.shape[0]  # [3, 3]

    # step 3: SVD decomposition
    U, S, Vt = torch.linalg.svd(C)  # C = U @ diag(S) @ Vt

    # step 4: rotation matrix
    R = U @ Vt  # [3, 3]

    # fix improper rotation (det(R) = -1 means reflection)
    # flip last column of V if needed to ensure proper rotation
    if torch.det(R) < 0:
        # create copy to avoid in-place modification (breaks autograd)
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] = -Vt_fixed[-1, :]
        R = U @ Vt_fixed

    # step 5: scale (optional)
    if with_scale:
        # scale = ratio of covariance trace to source variance
        var_src = (X ** 2).sum() / src.shape[0]
        s = (S.sum() / var_src).clamp(min=1e-8)  # clamp to avoid division by zero
    else:
        s = torch.tensor(1.0, device=src.device, dtype=src.dtype)

    # step 6: translation
    t = mu_dst.squeeze(0) - s * (R @ mu_src.squeeze(0))  # [3]

    return s, R, t


def _apply_rigid(pts: torch.Tensor, s: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    apply similarity transform: pts_aligned = s·R·pts + t

    handles both batched [B, N, 3] and unbatched [N, 3] inputs.
    """
    # pts @ R.t() is equivalent to (R @ pts.t()).t() but more efficient for batched ops
    return s * (pts @ R.t()) + t


# ============================================================
# loss function components
# ============================================================

def _bone_dirs(joints: torch.Tensor, bones: list) -> torch.Tensor:
    """
    compute normalized bone direction vectors (scale-invariant).

    for each bone (parent_idx, child_idx):
      direction = (child - parent) / ||child - parent||

    this is used for bone direction loss: ||D(MANO_bones) - D(MP_bones)||²
    advantages:
      - scale-invariant (robust to depth ambiguity in monocular vision)
      - constrains articulation even when absolute joint positions are noisy
      - helps prevent kinematically invalid poses (e.g., fingers bending wrong way)

    args:
      joints: [N, 3] joint positions
      bones: list of (parent_idx, child_idx) tuples

    returns:
      directions: [B, 3] normalized direction vectors (B = number of bones)
    """
    directions = []
    for i, j in bones:
        vec = joints[j] - joints[i]  # bone vector
        norm = torch.norm(vec) + 1e-8  # add epsilon to avoid division by zero
        directions.append(vec / norm)  # normalize to unit vector
    return torch.stack(directions, dim=0)  # [B, 3]


# ============================================================
# IK state (warm-start for temporal coherence)
# ============================================================

class _IKState:
    """
    persistent state for inverse kinematics across frames.

    warm-start strategy:
      - each frame starts optimization from previous frame's solution
      - this converts per-frame IK (hard) into tracking (easier)
      - drastically reduces iterations needed for convergence
      - provides temporal smoothness "for free"

    optimizer reuse:
      - creating new Adam optimizer every frame wastes ~10-15% time
      - we keep one optimizer and just update its parameters
    """
    def __init__(self, device: torch.device):
        # pose parameters θ ∈ R^45 (15 joints × 3 DoF axis-angle)
        self.pose = torch.zeros(1, 45, device=device, dtype=torch.float32)

        # shape parameters β ∈ R^10 (hand morphology)
        # kept fixed at zero for now (average hand shape)
        # future: learn per-user β during calibration phase
        self.betas = torch.zeros(1, 10, device=device, dtype=torch.float32)

        # adam optimizer (reused across frames for efficiency)
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # track if we've seen at least one frame (for smoothness loss)
        self.initialized = False

        # diagnostic: count frames processed
        self.frame_count = 0


# global singleton state (one instance for entire session)
_ik_state: Optional[_IKState] = None


def _get_state() -> _IKState:
    """get or create the global IK state singleton."""
    global _ik_state
    if _ik_state is None:
        _ik_state = _IKState(_DEVICE)
    return _ik_state


# ============================================================
# simple version (neutral pose, no optimization)
# ============================================================

def mano_from_landmarks_simple(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    convert mediapipe landmarks to mano hand mesh (neutral pose, no optimization).

    this is the fast fallback mode:
      - ignores landmarks, returns neutral pose (θ=0, β=0)
      - no inverse kinematics, no fitting
      - ~0.5ms per frame (just one MANO forward pass)

    use cases:
      - testing/debugging
      - fallback when IK diverges
      - low-power devices

    args:
      landmarks: [21, 3] mediapipe hand landmarks (ignored in this version)

    returns:
      vertices: [778, 3] MANO mesh vertices (neutral pose)
      joints: [21, 3] MANO joint positions (neutral pose)
      theta: [45] MANO pose parameters (all zeros for neutral)
      ik_error: float, set to 999.0 to indicate fallback mode
    """
    layer = _lazy_load_layer()

    # neutral pose: all rotations = 0, default shape
    pose = torch.zeros(1, 45, device=_DEVICE, dtype=torch.float32)
    betas = torch.zeros(1, 10, device=_DEVICE, dtype=torch.float32)

    # run mano forward pass (θ, β → V, J)
    verts, joints = layer(pose, betas)

    # convert from pytorch tensors to numpy arrays
    # [0] removes batch dimension (we only process one hand at a time)
    verts_np = verts[0].detach().cpu().numpy().astype(np.float32)
    joints_np = joints[0].detach().cpu().numpy().astype(np.float32)
    theta_np = pose[0].detach().cpu().numpy().astype(np.float32)  # [45]

    return verts_np, joints_np, theta_np, 999.0  # high error indicates fallback


# ============================================================
# IK version (full optimization with fallback)
# ============================================================

def mano_from_landmarks(landmarks: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    convert mediapipe landmarks to mano hand mesh via inverse kinematics optimization.

    algorithm:
      1. initialize θ from previous frame (warm-start)
      2. for each optimization step:
         a. forward MANO: θ → J(θ)
         b. pick corresponding joints via mapping Π
         c. solve for rigid alignment: (s,R,t) = umeyama(J_Π, X)
         d. compute losses:
            - L_joint = ||sRJ_Π + t - X||²_F        (joint position after alignment)
            - L_bone  = ||D(sRJ_Π) - D(X)||²_F      (bone directions, scale-invariant)
            - L_smooth = ||θ - θ_{t-1}||²           (temporal smoothness)
            - L_reg = ||θ||²                         (pose regularization)
         e. backprop and adam step
      3. final forward pass with optimized θ
      4. apply (s,R,t) to vertices for rendering
      5. if loss > threshold, fall back to simple mode

    mathematical formulation:
      minimize over θ ∈ R^45:
        L(θ) = λ_joint · L_joint(θ)
             + λ_bone · L_bone(θ)
             + λ_smooth · L_smooth(θ)
             + λ_reg · L_reg(θ)

      where (s,R,t) = argmin_{s,R,t} ||sRJ_Π(θ) + t1ᵀ - X||²_F  (closed-form via SVD)

    why this works:
      - umeyama removes global pose ambiguity (focus on articulation)
      - bone directions help when depth is ambiguous (monocular)
      - temporal smoothness kills jitter (warm-start tracking)
      - regularization prevents anatomically impossible poses
      - autodiff (pytorch) makes optimization fast (~1-4ms/frame)

    current limitations (V1):
      - MANO forward returns V_shaped (neutral), not full LBS
      - articulation comes from rigid alignment, not pose blend shapes
      - future V2: add rodrigues → FK → posedirs → skinning

    args:
      landmarks: [21, 3] mediapipe landmarks (preferably world coords in meters)
      verbose: if True, print loss values (for debugging)

    returns:
      vertices: [778, 3] MANO mesh vertices (aligned to camera space)
      joints: [21, 3] MANO joint positions (aligned to camera space)
      theta: [45] MANO pose parameters (axis-angle joint rotations)
      ik_error: float, final optimization loss (convergence quality metric)
    """
    layer = _lazy_load_layer()
    state = _get_state()

    # convert landmarks to torch tensor
    mp_landmarks = torch.tensor(landmarks, device=_DEVICE, dtype=torch.float32)

    # joint correspondence: mediapipe indices → MANO indices
    mp2mano = torch.tensor(_MANO21_FOR_MP, device=_DEVICE, dtype=torch.long)

    # initialize pose from warm-start (or zeros for first frame)
    pose = state.pose.clone().detach().requires_grad_(True)
    betas = state.betas.clone().detach()  # keep shape fixed for now
    betas.requires_grad_(False)

    # setup or reuse optimizer
    if state.optimizer is None:
        state.optimizer = torch.optim.Adam([pose], lr=LEARNING_RATE)
    else:
        # update optimizer to track new pose tensor
        state.optimizer.param_groups[0]['params'] = [pose]
        state.optimizer.param_groups[0]['lr'] = LEARNING_RATE

    # save previous pose for smoothness loss
    last_pose = state.pose.detach()

    # optimization loop
    for step in range(IK_STEPS):
        # forward pass: θ → V(θ), J(θ)
        verts, joints = layer(pose, betas)
        joints = joints[0]  # remove batch dimension: [1, 21, 3] → [21, 3]

        # select corresponding MANO joints (permutation Π)
        mano_sel = joints[mp2mano]  # [21, 3]

        # solve for rigid alignment: (s,R,t) = argmin ||sRJ + t - X||²
        s, R, t = _umeyama(mano_sel, mp_landmarks, with_scale=True)

        # apply rigid transform to MANO joints
        mano_aligned = _apply_rigid(mano_sel, s, R, t)

        # loss 1: joint position error (L2 after alignment)
        L_joint = ((mano_aligned - mp_landmarks) ** 2).mean()

        # loss 2: bone direction error (scale-invariant)
        mp_dirs = _bone_dirs(mp_landmarks, _MP_BONES)
        mano_dirs = _bone_dirs(mano_aligned, _MP_BONES)
        L_bone = ((mp_dirs - mano_dirs) ** 2).mean()

        # loss 3: temporal smoothness (stay close to previous frame)
        L_smooth = ((pose - last_pose) ** 2).mean()

        # loss 4: pose regularization (prefer small angles)
        L_reg = (pose ** 2).mean()

        # total loss (weighted sum)
        loss = (WEIGHT_JOINT * L_joint +
                WEIGHT_BONE * L_bone +
                WEIGHT_SMOOTH * L_smooth +
                WEIGHT_REG * L_reg)

        # backprop and optimize
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        state.optimizer.step()

    # verbose logging (optional)
    if verbose:
        print(f"IK frame {state.frame_count}: "
              f"L_joint={L_joint.item():.4f}, L_bone={L_bone.item():.4f}, "
              f"L_smooth={L_smooth.item():.4f}, L_reg={L_reg.item():.4f}, "
              f"total={loss.item():.4f}")

    # check convergence: if loss too high, fall back to simple mode
    if loss.item() > FALLBACK_THRESHOLD or torch.isnan(loss):
        if verbose:
            print(f"IK diverged (loss={loss.item():.4f}), falling back to simple mode")
        return mano_from_landmarks_simple(landmarks)

    # final forward pass with optimized pose
    verts, joints = layer(pose, betas)
    verts = verts[0]   # [778, 3]
    joints = joints[0]  # [21, 3]

    # apply rigid alignment to full mesh (so render matches camera space)
    mano_sel = joints[mp2mano]
    s, R, t = _umeyama(mano_sel, mp_landmarks, with_scale=True)
    verts_aligned = _apply_rigid(verts, s, R, t)
    joints_aligned = _apply_rigid(joints, s, R, t)
    mano_aligned = _apply_rigid(mano_sel, s, R, t)  # aligned selected joints for error calc

    # compute actual joint position error in millimeters
    # mean distance between aligned mano joints and mediapipe landmarks
    joint_errors = torch.norm(mano_aligned - mp_landmarks, dim=1)  # [21] distances
    ik_error_mm = joint_errors.mean().item() * 1000  # convert meters to mm

    # update warm-start state for next frame
    state.pose = pose.detach()
    state.betas = betas.detach()
    state.initialized = True
    state.frame_count += 1

    # convert to numpy
    verts_np = verts_aligned.detach().cpu().numpy().astype(np.float32)
    joints_np = joints_aligned.detach().cpu().numpy().astype(np.float32)
    theta_np = pose[0].detach().cpu().numpy().astype(np.float32)  # [45]

    return verts_np, joints_np, theta_np, ik_error_mm


# ============================================================
# debug utilities
# ============================================================

def debug_print_joint_order():
    """
    print MANO 21-joint positions in neutral pose for validating joint mapping.

    use this to refine _MANO21_FOR_MP:
      1. run this function once
      2. visually inspect xyz positions (wrist near origin, fingers extending)
      3. compare with mediapipe landmark order
      4. adjust _MANO21_FOR_MP indices so anatomical chains match

    mediapipe order (for reference):
      0: wrist
      1-4: thumb (CMC, MCP, IP, tip)
      5-8: index (MCP, PIP, DIP, tip)
      9-12: middle (MCP, PIP, DIP, tip)
      13-16: ring (MCP, PIP, DIP, tip)
      17-20: pinky (MCP, PIP, DIP, tip)
    """
    layer = _lazy_load_layer()

    # neutral pose
    pose = torch.zeros(1, 45, device=_DEVICE)
    betas = torch.zeros(1, 10, device=_DEVICE)

    # forward pass
    _, joints = layer(pose, betas)
    J = joints[0].detach().cpu().numpy()

    print("\n" + "=" * 60)
    print(" MANO 21-joint order (neutral pose)")
    print("=" * 60)
    print("index |       x       |       y       |       z       ")
    print("------+---------------+---------------+---------------")
    for i, j in enumerate(J):
        print(f"  {i:2d}  | {j[0]:+12.6f} | {j[1]:+12.6f} | {j[2]:+12.6f}")
    print("=" * 60)
    print("\nTIP: compare positions with mediapipe landmarks")
    print("     adjust _MANO21_FOR_MP so finger chains align anatomically")
    print("     wrist (0) should be near origin")
    print("     fingertips (16-20) should be farthest from wrist\n")
