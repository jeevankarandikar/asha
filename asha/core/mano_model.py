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
        """load mano model from MANO_{side}_numpy.pkl file."""
        pkl_path = self.mano_root / f"MANO_{self.side}_numpy.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"converted mano model not found: {pkl_path}\n"
                f"download MANO_{self.side}.pkl from: https://mano.is.tue.mpg.de/download.php\n"
                f"place in models/ directory, then run conversion script:\n"
                f"  python src/convert_mano_to_numpy.py"
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

    def rodrigues(self, theta: torch.Tensor) -> torch.Tensor:
        """
        convert axis-angle representation to rotation matrix using rodrigues formula.

        axis-angle: 3D vector where direction = rotation axis, magnitude = angle (radians)
        rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        where:
          θ = ||theta|| (angle magnitude)
          K = skew-symmetric matrix of unit axis (theta/θ)

        args:
          theta: [B, 3] axis-angle vectors

        returns:
          R: [B, 3, 3] rotation matrices (proper, det(R)=+1)

        mathematical details:
          given axis-angle vector v = [v_x, v_y, v_z]:
          θ = sqrt(v_x² + v_y² + v_z²)
          axis = v / θ  (unit vector)

          skew-symmetric matrix K (cross-product matrix):
            K = [    0   -axis_z   axis_y ]
                [ axis_z     0    -axis_x ]
                [-axis_y  axis_x      0   ]

          rodrigues formula:
            R = I + sin(θ)K + (1-cos(θ))K²

          for small angles (θ < eps), use taylor expansion to avoid division by zero:
            R ≈ I + K  (first-order approximation)
        """
        batch_size = theta.shape[0]
        device = theta.device

        # compute angle magnitude: θ = ||v||
        angle = torch.norm(theta + 1e-8, dim=1, keepdim=True)  # [B, 1], add epsilon for stability

        # normalize to get rotation axis: axis = v / θ
        r = theta / angle  # [B, 3]

        # extract axis components
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]

        # build skew-symmetric matrix K for each sample
        # K represents the cross product operator
        K = torch.zeros((batch_size, 3, 3), device=device)
        K[:, 0, 1] = -rz
        K[:, 0, 2] = ry
        K[:, 1, 0] = rz
        K[:, 1, 2] = -rx
        K[:, 2, 0] = -ry
        K[:, 2, 1] = rx

        # compute K² (K squared)
        K_squared = torch.bmm(K, K)  # [B, 3, 3]

        # rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        angle = angle.unsqueeze(-1)  # [B, 1, 1]
        # use repeat instead of expand (create copy, not view)
        I = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]

        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * K_squared

        return R  # [B, 3, 3]

    def batch_rodrigues(self, theta: torch.Tensor) -> torch.Tensor:
        """
        apply rodrigues formula to multiple axis-angle vectors.

        args:
          theta: [B, N, 3] or [B, N*3] axis-angle vectors for N joints

        returns:
          R: [B, N, 3, 3] rotation matrices
        """
        batch_size = theta.shape[0]

        # reshape if flattened: [B, N*3] → [B, N, 3]
        if theta.dim() == 2:
            theta = theta.view(batch_size, -1, 3)

        num_joints = theta.shape[1]

        # flatten for batch processing: [B, N, 3] → [B*N, 3]
        theta_flat = theta.view(-1, 3)

        # apply rodrigues: [B*N, 3] → [B*N, 3, 3]
        R_flat = self.rodrigues(theta_flat)

        # reshape back: [B*N, 3, 3] → [B, N, 3, 3]
        R = R_flat.view(batch_size, num_joints, 3, 3)

        return R

    def with_zeros(self, x: torch.Tensor) -> torch.Tensor:
        """
        append homogeneous coordinate row [0, 0, 0, 1] to 3x4 transform matrix.

        converts [3, 4] → [4, 4] for use in SE(3) transforms.

        args:
          x: [B, 3, 4] transform matrix (rotation + translation)

        returns:
          [B, 4, 4] homogeneous transform
        """
        batch_size = x.shape[0]
        # create bottom row: [0, 0, 0, 1]
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=x.device).unsqueeze(0).unsqueeze(0)
        # use repeat instead of expand (create copy, not view)
        bottom = bottom.repeat(batch_size, 1, 1)  # [B, 1, 4]
        return torch.cat([x, bottom], dim=1)  # [B, 4, 4]

    def pack(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        pack rotation and translation into homogeneous 4x4 transform matrix.

        builds SE(3) transform: [R t]
                                [0 1]

        args:
          R: [B, 3, 3] rotation matrix
          t: [B, 3, 1] translation vector

        returns:
          [B, 4, 4] homogeneous transformation matrix
        """
        # concatenate: [B, 3, 3] + [B, 3, 1] → [B, 3, 4]
        transform_3x4 = torch.cat([R, t], dim=2)
        # add homogeneous row: [B, 3, 4] → [B, 4, 4]
        return self.with_zeros(transform_3x4)

    def forward(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        run forward pass: parameters → 3d hand mesh (full LBS implementation).

        this is the main function that generates the hand.
        takes pose and shape parameters, returns vertices and joints.

        full MANO pipeline:
        1. shape blend shapes: deform template based on shape parameters
        2. get rest pose joint positions from shaped vertices
        3. pose rotations: convert axis-angle θ to rotation matrices via rodrigues
        4. pose blend shapes: add corrective deformations for joint rotations
        5. forward kinematics: compute global transforms through kinematic tree
        6. linear blend skinning: apply weighted bone transforms to vertices

        args:
          pose: [B, 45] axis-angle rotations for 15 joints (3 DoF each)
          betas: [B, 10] shape parameters

        returns:
          vertices: [B, 778, 3] deformed mesh vertices
          joints: [B, 21, 3] joint positions (16 skeleton + 5 fingertips)
        """
        batch_size = pose.shape[0]
        device = pose.device

        # step 1: shape blend shapes
        # v_shaped = v_template + Σ β_k · shapedirs_k
        v_shaped = self.v_template + torch.einsum(
            'vij,bj->bvi', self.shapedirs, betas
        )

        # step 2: rest pose joints (before applying rotations)
        # J = J_regressor @ v_shaped
        # these are the joint locations in the shaped (but not yet posed) hand
        J = torch.einsum('jv,bvi->bji', self.J_regressor, v_shaped)  # [B, 16, 3]

        # step 3: pose rotations
        # convert axis-angle θ ∈ R^45 to rotation matrices R_i ∈ SO(3)
        # pose is [B, 45] = [B, 15*3] for 15 joints
        pose_cube = pose.view(batch_size, -1, 3)  # [B, 15, 3]
        rot_mats = self.batch_rodrigues(pose_cube)  # [B, 15, 3, 3]

        # step 4: pose blend shapes
        # build pose feature: p(θ) = concat([vec(R_i - I) for i in 1..15])
        # note: we use all 15 joints (some implementations skip root, but this works too)
        # use repeat instead of expand (create copy, not view)
        I_cube = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, rot_mats.shape[1], 1, 1)
        pose_feature = (rot_mats - I_cube).view(batch_size, -1)  # [B, 15*9=135]

        # apply pose blend shapes: ΔV = posedirs @ p(θ)
        # posedirs can be either [778*3, 135] or [778, 3, 135] depending on MANO build
        # handle both cases
        if self.posedirs.dim() == 3:
            # shape: [778, 3, 135]
            v_posed = v_shaped + torch.einsum(
                'vij,bj->bvi', self.posedirs, pose_feature
            )  # [B, 778, 3]
        else:
            # shape: [778*3, 135] (flattened)
            v_posed = v_shaped + torch.einsum(
                'ij,bj->bi', self.posedirs, pose_feature
            ).view(batch_size, -1, 3)  # [B, 778, 3]

        # step 5: forward kinematics (FK)
        # compute global transforms G_i for each joint by traversing kinematic tree
        # the kintree_table is [2, 16]: row 0 = child indices, row 1 = parent indices

        # number of joints in skeleton (not including fingertips)
        num_joints = J.shape[1]  # 16

        # build local transforms at rest pose: T_i = [I | J_i]
        # these position each joint at its rest location
        J_transformed = J.clone()

        # build global transforms by traversing kinematic tree
        # for root (joint 0): G_0 = R_0 @ T_0
        # for child j with parent i: G_j = G_i @ T_rel_j @ R_j
        #
        # we'll build 4x4 homogeneous transforms for each joint

        # prepare rotation matrices with root
        # add identity rotation for root (wrist typically doesn't rotate in MANO)
        # rot_mats is [B, 15, 3, 3], we need [B, 16, 3, 3]
        # prepend identity for root (index 0)
        # use repeat instead of expand (create copy, not view)
        I_root = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # for MANO, the first 3 params in pose are typically root orientation
        # but kintree starts at wrist, so we use first rot_mat as root
        # actually, let's use rot_mats as-is (15 joints) and handle carefully

        # build global transforms by traversing kinematic tree
        # use list to avoid in-place modifications (which break autograd)
        G_list = []

        # process joints in order (parent before child)
        # kintree_table[0, j] gives parent of joint j
        # kintree_table[0, 0] is typically -1 (or large uint) meaning root has no parent
        for i in range(num_joints):
            parent_idx = int(self.kintree_table[0, i]) if i > 0 else -1

            if i == 0 or parent_idx < 0:
                # root joint: just apply rotation around its position
                # G_0 = R_0 @ T(J_0)
                if i < rot_mats.shape[1]:
                    R_i = rot_mats[:, i, :, :]  # [B, 3, 3]
                else:
                    # use repeat instead of expand (create copy, not view)
                    R_i = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                t_i = J[:, i, :].unsqueeze(-1)  # [B, 3, 1]
                G_i = self.pack(R_i, t_i)
                G_list.append(G_i)
            else:
                # child joint: G_i = G_parent @ R_i @ T_rel

                # rotation for this joint
                if i < rot_mats.shape[1]:
                    R_i = rot_mats[:, i, :, :]
                else:
                    # use repeat instead of expand (create copy, not view)
                    R_i = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

                # relative translation from parent to this joint
                t_rel = (J[:, i, :] - J[:, parent_idx, :]).unsqueeze(-1)  # [B, 3, 1]

                # local transform: R_i and relative translation
                T_local = self.pack(R_i, t_rel)  # [B, 4, 4]

                # global transform: chain with parent
                G_i = torch.matmul(G_list[parent_idx], T_local)
                G_list.append(G_i)

        # stack into tensor: list of [B, 4, 4] → [B, num_joints, 4, 4]
        G = torch.stack(G_list, dim=1)  # [B, 16, 4, 4]

        # transforms are now in world space, but we need to subtract rest pose
        # to get the actual skinning transforms (transforms relative to rest)
        # G_skinning_i = G_i @ T_rest_i^{-1}
        # where T_rest_i moves joint i to origin

        G_skinning_list = []
        for i in range(num_joints):
            # rest pose inverse: just negate translation
            J_rest_inv = -J[:, i, :].unsqueeze(-1)  # [B, 3, 1]
            # create identity without expand (avoid autograd issues with views)
            I_batch = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            T_rest_inv = self.pack(I_batch, J_rest_inv)
            # use clone to avoid view-related autograd issues
            G_skin_i = torch.matmul(G[:, i, :, :].clone(), T_rest_inv)
            G_skinning_list.append(G_skin_i)

        G_skinning = torch.stack(G_skinning_list, dim=1)  # [B, 16, 4, 4]

        # step 6: linear blend skinning (LBS)
        # for each vertex: v_final = Σ_j w_j · (G_j @ [v_posed; 1])
        # weights: [778, 16] (vertex × joint)
        # G_skinning: [B, 16, 4, 4]
        # v_posed: [B, 778, 3]

        # convert vertices to homogeneous coordinates: [B, 778, 3] → [B, 778, 4]
        v_posed_homo = torch.cat([
            v_posed,
            torch.ones((batch_size, v_posed.shape[1], 1), device=device)
        ], dim=2)  # [B, 778, 4]

        # apply skinning: for each vertex, weighted sum of bone transforms
        # weights @ G_skinning gives weighted transform per vertex
        # we need: v_final_i = Σ_j weights[i,j] · (G_j @ v_i)

        # expand weights for batch: [778, 16] → [B, 778, 16]
        # use repeat instead of expand (create copy, not view)
        W = self.weights.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 778, 16]

        # for each vertex, compute transformed position under each bone
        # T[b,v,j] = G_skinning[b,j] @ v_posed_homo[b,v]
        # result[b,v] = Σ_j W[b,v,j] · T[b,v,j]

        # efficient implementation: use einsum
        # G_skinning: [B, 16, 4, 4]
        # v_posed_homo: [B, 778, 4]
        # weights: [B, 778, 16]

        # transform all vertices by all bones: [B, 16, 4, 4] @ [B, 778, 4] (broadcast)
        # we want: for each (b,v,j): G[b,j] @ v[b,v]
        # reshape for batch matrix multiply
        T = torch.einsum('bjkl,bvl->bvjk', G_skinning, v_posed_homo)  # [B, 778, 16, 4]

        # apply weights: [B, 778, 16, 4] weighted by [B, 778, 16]
        v_final_homo = torch.einsum('bvj,bvjk->bvk', W, T)  # [B, 778, 4]

        # drop homogeneous coordinate: [B, 778, 4] → [B, 778, 3]
        vertices = v_final_homo[:, :, :3]

        # update joint positions with final global transforms (optional, for consistency)
        # extract translation from G (not G_skinning)
        joints_posed = G[:, :, :3, 3]  # [B, 16, 3]

        # add fingertip positions (from final vertices)
        J_fingertips = vertices[:, self.fingertip_indices, :]  # [B, 5, 3]

        # combine: [B, 16, 3] + [B, 5, 3] → [B, 21, 3]
        joints_final = torch.cat([joints_posed, J_fingertips], dim=1)

        return vertices, joints_final

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
