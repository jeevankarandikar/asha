"""
interface to mano parametric hand model. uses custom loader and apple silicon mps when available.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from mano_layer import CustomMANOLayer


def get_torch_device() -> torch.device:
    """pick best device: mps (apple silicon gpu) if available, otherwise cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# where mano model files live
_MANO_ROOT = Path(__file__).resolve().parents[1] / "mano_models"
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


def mano_from_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    convert mediapipe landmarks to mano hand mesh.

    currently just returns neutral pose (flat hand).
    the hard part (inverse kinematics) isn't implemented yet.

    how it should work:
    1. mediapipe gives us: where joints are (positions)
    2. mano needs: how joints are rotated (angles)
    3. problem: converting position → rotation is hard (inverse kinematics)
    4. solution: would need neural network or optimization

    for now: just ignores landmarks and returns neutral pose.
    """
    layer = _lazy_load_layer()

    # placeholder: neutral pose (all rotations = 0, default shape)
    # mano parameters:
    # - pose: 45 numbers (15 joints × 3 rotation angles each)
    # - betas: 10 numbers (hand shape variation - thin/thick, long/short fingers, etc)
    pose = torch.zeros(1, 45, device=_DEVICE, dtype=torch.float32)
    betas = torch.zeros(1, 10, device=_DEVICE, dtype=torch.float32)

    # run mano forward pass (parameters → 3d mesh)
    verts, joints = layer(pose, betas)

    # convert from pytorch tensors to numpy arrays
    # [0] removes batch dimension (we only process one hand at a time)
    verts_np = verts[0].detach().cpu().numpy().astype(np.float32)
    joints_np = joints[0].detach().cpu().numpy().astype(np.float32)

    return verts_np, joints_np
