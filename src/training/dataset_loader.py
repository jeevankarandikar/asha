"""
training dataset loader with curriculum mixing for freihand, ho-3d, and dexycb.

features:
  - curriculum mixing: start with 70% freihand, ramp to 50% mixed over 20 epochs
  - ho-3d loader (103k frames, occlusions)
  - dexycb subset loader (sample 100k frames for grasping diversity)
  - data augmentations: rotations, scaling, cutmix
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class FreiHANDDataset(Dataset):
    """freihand dataset loader for training."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[object] = None,
    ):
        """initialize freihand dataset."""
        self.root = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # load annotations
        self.xyz_path = self.root / "training_xyz.json"
        self.mano_path = self.root / "training_mano.json"
        self.K_path = self.root / "training_K.json"
        
        if not self.xyz_path.exists():
            raise FileNotFoundError(f"freihand annotations not found: {self.xyz_path}")
        
        # load 3d joint positions
        with open(self.xyz_path, 'r') as f:
            self.xyz = np.array(json.load(f))  # [32560, 21, 3]
        
        # load mano parameters
        if self.mano_path.exists():
            with open(self.mano_path, 'r') as f:
                mano_data = json.load(f)
                self.theta = np.array(mano_data.get('theta', []))  # [32560, 45]
                self.beta = np.array(mano_data.get('beta', []))  # [32560, 10]
        else:
            self.theta = None
            self.beta = None
        
        # image directory
        self.rgb_dir = self.root / "training" / "rgb"
        
        # build index: 32k unique poses × 4 views = 130k images
        self.indices = []
        for base_idx in range(32560):
            for view_idx in range(4):
                img_id = base_idx * 4 + view_idx
                img_path = self.rgb_dir / f"{img_id:08d}.jpg"
                if img_path.exists():
                    self.indices.append((base_idx, view_idx))
        
        # train/val split (90/10)
        if split == "train":
            self.indices = self.indices[:int(len(self.indices) * 0.9)]
        else:
            self.indices = self.indices[int(len(self.indices) * 0.9):]
        
        print(f"[freihand {split}] loaded {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        base_idx, view_idx = self.indices[idx]
        img_id = base_idx * 4 + view_idx
        
        # load image
        img_path = self.rgb_dir / f"{img_id:08d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            # fallback: create dummy image if read fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ground truth
        joints_3d = self.xyz[base_idx].astype(np.float32)  # [21, 3]
        theta = self.theta[base_idx].astype(np.float32) if self.theta is not None else np.zeros(45, dtype=np.float32)
        beta = self.beta[base_idx].astype(np.float32) if self.beta is not None else np.zeros(10, dtype=np.float32)
        
        # apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'theta': torch.from_numpy(theta),
            'beta': torch.from_numpy(beta),
            'joints_3d': torch.from_numpy(joints_3d),
        }


class HO3DDataset(Dataset):
    """ho-3d dataset loader for training (103k frames with occlusions)."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[object] = None,
        max_samples: Optional[int] = None,
    ):
        """initialize ho-3d dataset."""
        self.root = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"HO-3D split directory not found: {self.split_dir}")
        
        # Collect all sequences
        sequences = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        
        # Build frame index
        self.frame_index = []
        for seq_dir in sequences:
            rgb_dir = seq_dir / "rgb"
            if not rgb_dir.exists():
                continue
            
            frames = sorted(rgb_dir.glob("*.jpg"))
            for frame_path in frames:
                frame_num = int(frame_path.stem)
                self.frame_index.append((seq_dir, frame_num))
        
        # Limit samples if requested
        if max_samples is not None:
            self.frame_index = self.frame_index[:max_samples]
        
        print(f"[HO-3D {split}] Loaded {len(self.frame_index)} samples")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        seq_dir, frame_num = self.frame_index[idx]
        
        # Load RGB image
        img_path = seq_dir / "rgb" / f"{frame_num:04d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: create dummy image if read fails
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Try to load MANO parameters from meta
        theta = np.zeros(45, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        joints_3d = np.zeros((21, 3), dtype=np.float32)
        
        meta_path = seq_dir / "meta" / f"{frame_num:05d}.pkl"
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                
                if 'handPose' in meta:  # MANO theta
                    theta = np.array(meta['handPose'][:45], dtype=np.float32)
                if 'handBeta' in meta:  # MANO beta
                    beta = np.array(meta['handBeta'][:10], dtype=np.float32)
                if 'handJoints3D' in meta:
                    joints_3d = np.array(meta['handJoints3D'], dtype=np.float32)
            except Exception as e:
                pass  # Use zeros if loading fails
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'theta': torch.from_numpy(theta),
            'beta': torch.from_numpy(beta),
            'joints_3d': torch.from_numpy(joints_3d),
        }


class DexYCBDataset(Dataset):
    """DexYCB dataset loader (sample 100K frames for grasping diversity)."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[object] = None,
        max_samples: int = 100000,
    ):
        """
        Initialize DexYCB dataset subset.
        
        Args:
            dataset_path: path to DexYCB root directory
            split: 'train' or 'val'
            transform: albumentations transform pipeline
            max_samples: maximum number of samples (default: 100K)
        """
        self.root = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # DexYCB structure: sequences/ with subdirectories
        # For simplicity, we'll sample frames from available sequences
        # In practice, you'd load the full annotation file
        
        # Placeholder: build frame index from available sequences
        sequences_dir = self.root / "sequences"
        if not sequences_dir.exists():
            raise FileNotFoundError(f"DexYCB sequences directory not found: {sequences_dir}")
        
        sequences = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])
        
        self.frame_index = []
        for seq_dir in sequences:
            rgb_dir = seq_dir / "color"
            if not rgb_dir.exists():
                continue
            
            frames = sorted(rgb_dir.glob("*.jpg"))
            for frame_path in frames[:max_samples // len(sequences)]:  # Distribute samples
                frame_num = int(frame_path.stem)
                self.frame_index.append((seq_dir, frame_num))
        
        # Limit to max_samples
        if len(self.frame_index) > max_samples:
            self.frame_index = self.frame_index[:max_samples]
        
        print(f"[DexYCB {split}] Loaded {len(self.frame_index)} samples")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        seq_dir, frame_num = self.frame_index[idx]
        
        # Load RGB image
        img_path = seq_dir / "color" / f"{frame_num:06d}.jpg"
        if not img_path.exists():
            img_path = seq_dir / "color" / f"{frame_num:04d}.jpg"  # Try alternative format
        
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: create dummy image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Try to load MANO parameters (DexYCB format)
        theta = np.zeros(45, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        joints_3d = np.zeros((21, 3), dtype=np.float32)
        
        # DexYCB annotations are typically in separate files
        # This is a placeholder - adjust based on actual DexYCB structure
        meta_path = seq_dir / "meta" / f"{frame_num:06d}.pkl"
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                # Extract MANO parameters (adjust keys based on actual format)
                if 'hand_pose' in meta:
                    theta = np.array(meta['hand_pose'][:45], dtype=np.float32)
                if 'hand_beta' in meta:
                    beta = np.array(meta['hand_beta'][:10], dtype=np.float32)
            except Exception:
                pass
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'theta': torch.from_numpy(theta),
            'beta': torch.from_numpy(beta),
            'joints_3d': torch.from_numpy(joints_3d),
        }


class CurriculumMixedDataset(Dataset):
    """
    Curriculum mixing dataset: starts with mostly FreiHAND, ramps to mixed datasets.
    
    Strategy:
    - Initial epochs: 70% FreiHAND, 20% HO-3D, 10% DexYCB
    - Ramp over 20 epochs to: 50% FreiHAND, 30% HO-3D, 20% DexYCB
    - Prevents forgetting while introducing diversity
    """
    
    def __init__(
        self,
        freihand_dataset: Dataset,
        ho3d_dataset: Dataset,
        dexycb_dataset: Dataset,
        epoch: int = 0,
        ramp_epochs: int = 20,
    ):
        """
        Initialize curriculum mixed dataset.
        
        Args:
            freihand_dataset: FreiHAND dataset
            ho3d_dataset: HO-3D dataset
            dexycb_dataset: DexYCB dataset
            epoch: current training epoch
            ramp_epochs: number of epochs to ramp mixing ratio
        """
        self.freihand = freihand_dataset
        self.ho3d = ho3d_dataset
        self.dexycb = dexycb_dataset
        self.epoch = epoch
        self.ramp_epochs = ramp_epochs
        
        # Compute mixing ratios
        progress = min(epoch / ramp_epochs, 1.0)  # 0.0 → 1.0
        
        # Start: 70% FreiHAND, 20% HO-3D, 10% DexYCB
        # End: 50% FreiHAND, 30% HO-3D, 20% DexYCB
        self.freihand_ratio = 0.7 - 0.2 * progress
        self.ho3d_ratio = 0.2 + 0.1 * progress
        self.dexycb_ratio = 0.1 + 0.1 * progress
        
        # Build combined index
        self.indices = []
        
        # FreiHAND samples
        n_freihand = int(len(freihand_dataset) * self.freihand_ratio)
        self.indices.extend([('freihand', i) for i in range(n_freihand)])
        
        # HO-3D samples
        n_ho3d = int(len(ho3d_dataset) * self.ho3d_ratio)
        self.indices.extend([('ho3d', i) for i in range(n_ho3d)])
        
        # DexYCB samples
        n_dexycb = int(len(dexycb_dataset) * self.dexycb_ratio)
        self.indices.extend([('dexycb', i) for i in range(n_dexycb)])
        
        # Shuffle
        random.shuffle(self.indices)
        
        print(f"[Curriculum Mixed] Epoch {epoch}/{ramp_epochs}")
        print(f"  FreiHAND: {self.freihand_ratio:.1%} ({n_freihand} samples)")
        print(f"  HO-3D: {self.ho3d_ratio:.1%} ({n_ho3d} samples)")
        print(f"  DexYCB: {self.dexycb_ratio:.1%} ({n_dexycb} samples)")
        print(f"  Total: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        dataset_name, sample_idx = self.indices[idx]
        
        if dataset_name == 'freihand':
            return self.freihand[sample_idx]
        elif dataset_name == 'ho3d':
            return self.ho3d[sample_idx]
        else:  # dexycb
            return self.dexycb[sample_idx]
    
    def update_epoch(self, epoch: int):
        """update epoch for curriculum mixing."""
        self.epoch = epoch
        # Rebuild indices with new ratios
        self.__init__(
            self.freihand,
            self.ho3d,
            self.dexycb,
            epoch,
            self.ramp_epochs,
        )


def get_train_transforms(augment: bool = True, image_size: int = 224):
    """
    get training data augmentation pipeline.

    augmentations:
      - random rotations ±30°
      - scaling 0.8-1.2
      - cutmix for synthetic occlusions (mimics dexycb/ho-3d)
    """
    if not augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=30, p=0.5),  # ±30° rotations
        A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),  # Scaling 0.8-1.2
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),  # CutMix-like occlusions
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: target image size (default 224)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def create_dataloaders(
    freihand_path: str,
    ho3d_path: Optional[str] = None,
    dexycb_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    epoch: int = 0,
    use_curriculum: bool = True,
    ramp_epochs: int = 20,
    image_size: int = 224,
):
    """
    create training and validation dataloaders.
    
    returns train_loader, val_loader pytorch dataloaders.
    """
    # training transforms
    train_transform = get_train_transforms(augment=True, image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)
    
    # FreiHAND datasets
    train_freihand = FreiHANDDataset(freihand_path, split="train", transform=train_transform)
    val_freihand = FreiHANDDataset(freihand_path, split="val", transform=val_transform)
    
    # HO-3D dataset (if available)
    if ho3d_path and Path(ho3d_path).exists():
        train_ho3d = HO3DDataset(ho3d_path, split="train", transform=train_transform)
    else:
        train_ho3d = None
    
    # DexYCB dataset (if available)
    if dexycb_path and Path(dexycb_path).exists():
        train_dexycb = DexYCBDataset(dexycb_path, split="train", transform=train_transform, max_samples=100000)
    else:
        train_dexycb = None
    
    # Create training dataset
    if use_curriculum and train_ho3d is not None and train_dexycb is not None:
        train_dataset = CurriculumMixedDataset(
            train_freihand,
            train_ho3d,
            train_dexycb,
            epoch=epoch,
            ramp_epochs=ramp_epochs,
        )
    else:
        # Fallback to FreiHAND only
        train_dataset = train_freihand
    
    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_freihand,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
