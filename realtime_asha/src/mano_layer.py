"""
custom mano loader that reads converted numpy pkl files. avoids chumpy/python 3.11 issues.
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any


class CustomMANOLayer:
    """
    custom mano model loader.

    loads converted pkl files (created by convert_mano_to_numpy.py).
    implements forward pass: parameters → 3d hand mesh.
    """

    def __init__(self, mano_root: str, side: str = "right", device: str = "cpu"):
        """initialize mano model. loads pkl files and sets up pytorch tensors."""
        self.mano_root = Path(mano_root)
        self.side = side.upper()  # RIGHT or LEFT
        self.device = device

        # load the converted mano pkl file
        self.data = self._load_mano_model()

        # convert numpy arrays to pytorch tensors
        # move to device (cpu or mps)
        self._setup_tensors()

    def _load_mano_model(self) -> Dict[str, Any]:
        """load mano model from MANO_{side}_CONVERTED.pkl file."""
        pkl_path = self.mano_root / f"MANO_{self.side}_CONVERTED.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"converted mano model not found: {pkl_path}\n"
                f"run conversion script first:\n"
                f"  python realtime_asha/src/convert_mano_to_numpy.py"
            )

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def _setup_tensors(self):
        """
        convert all numpy arrays to pytorch tensors.

        loads various matrices that define how the hand deforms:
        - shape blend shapes: change hand proportions (thin/thick, long/short)
        - pose blend shapes: realistic deformations when joints bend
        - template: base hand shape at rest
        - joint regressor: how to find joint positions from vertex positions
        - weights: how much each bone affects each vertex
        """

        # shape blend shapes (10 parameters control hand proportions)
        # each parameter deforms the hand in a specific way
        self.shapedirs = torch.from_numpy(
            self.data['shapedirs'].astype(np.float32)
        ).to(self.device)

        # pose blend shapes (how skin deforms when joints rotate)
        # makes knuckles bulge, skin wrinkle, etc
        self.posedirs = torch.from_numpy(
            self.data['posedirs'].astype(np.float32)
        ).to(self.device)

        # template vertices (778 points that make up neutral hand)
        self.v_template = torch.from_numpy(
            self.data['v_template'].astype(np.float32)
        ).to(self.device)

        # joint regressor (finds 16 skeleton joint positions from vertices)
        # weighted average of nearby vertices
        self.J_regressor = torch.from_numpy(
            self.data['J_regressor'].toarray().astype(np.float32)
        ).to(self.device)

        # skinning weights (how much each bone affects each vertex)
        # vertices near a joint move more with that joint
        self.weights = torch.from_numpy(
            self.data['weights'].astype(np.float32)
        ).to(self.device)

        # kinematic tree (parent-child relationships between bones)
        self.kintree_table = self.data['kintree_table']

        # mesh faces (which vertices connect to form triangles)
        # defines the surface of the hand
        self.faces = torch.from_numpy(
            self.data['f'].astype(np.int64)
        ).to(self.device)

        # optional: pca components for hand pose
        # compact representation of common hand poses
        if 'hands_components' in self.data:
            self.hands_components = torch.from_numpy(
                self.data['hands_components'].astype(np.float32)
            ).to(self.device)
        else:
            self.hands_components = None

        # optional: mean pose
        if 'hands_mean' in self.data:
            self.hands_mean = torch.from_numpy(
                self.data['hands_mean'].astype(np.float32)
            ).to(self.device)
        else:
            self.hands_mean = torch.zeros(45, dtype=torch.float32).to(self.device)

        # fingertip vertices
        # mano gives us 16 skeleton joints, but mediapipe outputs 21
        # we add 5 fingertip points from specific vertices
        self.fingertip_indices = torch.tensor(
            [745, 317, 444, 556, 673],  # thumb, index, middle, ring, pinky tips
            dtype=torch.long,
            device=self.device
        )

    def forward(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        run forward pass: parameters → 3d hand mesh.

        this is the main function that generates the hand.
        takes pose and shape parameters, returns vertices and joints.

        mano model steps:
        1. start with template hand (neutral pose)
        2. apply shape parameters (make hand thin/thick, fingers long/short)
        3. find joint positions
        4. apply pose parameters (rotate joints - not implemented yet)
        5. return final mesh

        currently simplified - just does shape, skips pose rotations.
        """
        batch_size = pose.shape[0]

        # step 1: apply shape blend shapes
        # deforms template based on shape parameters (betas)
        # v_shaped = template + sum(shape_parameter_i * shape_blend_shape_i)
        v_shaped = self.v_template + torch.einsum(
            'vij,bj->bvi', self.shapedirs, betas
        )

        # step 2: get skeleton joint positions (16 joints)
        # weighted average of vertex positions
        # this finds where wrist, knuckles, finger joints are
        J_skeleton = torch.einsum('jv,bvi->bji', self.J_regressor, v_shaped)

        # step 3: get fingertip positions (5 fingertips)
        # take specific vertices that are at finger tips
        J_fingertips = v_shaped[:, self.fingertip_indices, :]

        # step 4: combine to get all 21 joints
        # now we match mediapipe's 21-point output
        joints = torch.cat([J_skeleton, J_fingertips], dim=1)

        # step 5: apply pose (not implemented)
        # full mano would:
        # - convert pose parameters to rotation matrices
        # - rotate each joint
        # - propagate rotations through kinematic tree
        # - apply skinning weights to vertices
        # for now: just use shaped vertices as-is (neutral pose)
        v_posed = v_shaped

        # return mesh vertices and joint positions
        vertices = v_posed

        return vertices, joints

    def __call__(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """allow calling instance as function: mano_layer(pose, betas) same as mano_layer.forward(pose, betas)."""
        return self.forward(pose, betas)


def load_mano_layer(mano_root: str, side: str, device: torch.device) -> CustomMANOLayer:
    """helper to load mano layer. wraps CustomMANOLayer constructor."""
    return CustomMANOLayer(
        mano_root=mano_root,
        side=side,
        device=str(device)
    )
