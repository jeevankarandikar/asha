# Dataset Download Guide

Complete guide for downloading FreiHAND and HO-3D datasets for Experiment 5.

---

## FreiHAND Dataset

### Option 1: Automatic Download (Recommended)

```bash
./scripts/download_datasets.sh freihand
```

Downloads to: `~/datasets/freihand/`

### Option 2: Direct Download (Official)

**Training Set v2** (3.7 GB):
```bash
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
```

**Evaluation Set with Annotations** (724 MB):
```bash
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip
```

Then extract:
```bash
mkdir -p ~/datasets/freihand
cd ~/datasets/freihand
unzip FreiHAND_pub_v2.zip
unzip FreiHAND_pub_v2_eval.zip
```

### Option 3: Kaggle (Alternative)

**Link**: https://www.kaggle.com/datasets/danieldelro/freihand

**Via Kaggle CLI**:
```bash
pip install kaggle
kaggle datasets download -d danieldelro/freihand -p ~/datasets/freihand
cd ~/datasets/freihand && unzip freihand.zip
```

**Via Browser**:
1. Go to https://www.kaggle.com/datasets/danieldelro/freihand
2. Click "Download" (requires Kaggle account)
3. Extract to `~/datasets/freihand/`

### FreiHAND Dataset Structure

After extraction:
```
~/datasets/freihand/
├── training/
│   ├── rgb/              # 130,240 images (32,560 unique × 4 views)
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   └── ...
│   └── mask/             # Segmentation masks
├── evaluation/
│   └── rgb/              # 3,960 images
├── training_xyz.json     # [32560, 21, 3] 3D joint positions
├── training_mano.json    # MANO parameters (shape + pose)
└── training_K.json       # [32560, 3, 3] Camera intrinsics
```

### FreiHAND Dataset Info

- **Size**: 4.4 GB total
- **Images**: 134,200 total (130,240 train + 3,960 eval)
- **Unique samples**: 32,560 training samples
- **Views per sample**: 4 different camera angles
- **Ground truth**: Multi-view triangulation (~5mm accuracy)
- **Resolution**: 224×224 pixels
- **Paper**: FreiHAND (ICCV 2019)

---

## HO-3D Dataset (Hand-Object 3D)

### Option 1: Kaggle (Easiest)

**Link**: https://www.kaggle.com/datasets/marcmarais/ho3d-v3

**Via Kaggle CLI**:
```bash
pip install kaggle
kaggle datasets download -d marcmarais/ho3d-v3 -p ~/datasets/ho3d
cd ~/datasets/ho3d && unzip ho3d-v3.zip
```

**Via Browser**:
1. Go to https://www.kaggle.com/datasets/marcmarais/ho3d-v3
2. Click "Download" (requires Kaggle account)
3. Extract to `~/datasets/ho3d/`

### Option 2: Official OneDrive

**HO-3D v2**:
- Link: https://1drv.ms/f/c/11742dd40d1cbdc1/ElPb2rhOCeRMg-dFSM3iwO8B5nS1SgnQJs9F6l28G0pKKg?e=TMuxgr
- Download via browser, extract to `~/datasets/ho3d/`

**HO-3D v3** (Latest):
- Link: https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO
- Download via browser, extract to `~/datasets/ho3d/`

**Ground Truth Annotations** (Released Nov 3, 2024):
- v2 GT: https://1drv.ms/f/c/11742dd40d1cbdc1/EjSh5wMqNilPoQKA0nQF2NMBl0rfyg1gZQCo0k3iXv8vig
- v3 GT: https://1drv.ms/f/c/11742dd40d1cbdc1/EqzdBm7UDWVCmdxCyP373eQBhia924vXa4i85WqvWNLHYg?e=k6kshD

### Option 3: GitHub Repository

```bash
git clone https://github.com/shreyashampali/ho3d.git ~/datasets/ho3d/ho3d_repo
cd ~/datasets/ho3d/ho3d_repo
# Follow README instructions for dataset access
```

### HO-3D Dataset Structure

After extraction:
```
~/datasets/ho3d/
├── train/
│   ├── ABF10/           # Sequence folders
│   │   ├── rgb/         # RGB images
│   │   ├── depth/       # Depth maps
│   │   └── meta/        # Annotations
│   ├── BB10/
│   ├── GPMF10/
│   └── ...              # 10 sequences total
├── evaluation/
│   └── ...              # Evaluation sequences
└── models/
    └── ...              # YCB object models
```

### HO-3D Dataset Info

- **Version**: v3 (latest, released 2024)
- **Images**: 103,462 annotated images
- **Sequences**: 10 training sequences
- **Objects**: YCB objects (mustard bottle, power drill, etc.)
- **Ground truth**: RGB-D capture with object tracking
- **Resolution**: 640×480 pixels
- **Paper**: HOnnotate (CVPR 2020)

---

## Kaggle Setup (If Using Kaggle Downloads)

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Get API Credentials

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

### 3. Setup Credentials

**Linux/Mac**:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows**:
```
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### 4. Download Datasets

```bash
# FreiHAND
kaggle datasets download -d danieldelro/freihand -p ~/datasets/freihand

# HO-3D v3
kaggle datasets download -d marcmarais/ho3d-v3 -p ~/datasets/ho3d
```

---

## Verify Downloads

### Test FreiHAND

```bash
python src/experiments/freihand_loader.py \
  --dataset-path ~/datasets/freihand \
  --num-samples 10
```

Expected output:
```
[FreiHAND] Loading from: /Users/you/datasets/freihand
  → Loaded 32560 samples
  → 3D joints: (32560, 21, 3) (samples, 21 joints, xyz)

[FreiHAND Evaluation]
  Samples: 10
  Detection rate: 90.0%
  Mean IK error: 12.34 mm
```

### Test HO-3D

```bash
# TODO: Create HO-3D loader similar to freihand_loader.py
# For now, just verify files exist:
ls ~/datasets/ho3d/train/
```

Expected:
```
ABF10  ABF11  ABF12  ABF13  ABF14
BB10   BB11   BB12   BB13   BB14
```

---

## Troubleshooting

### FreiHAND Issues

**"Annotations not found"**:
```bash
# Check if files exist
ls ~/datasets/freihand/*.json

# If missing, redownload evaluation set
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip
unzip FreiHAND_pub_v2_eval.zip
```

**"No such file or directory"**:
```bash
# Check extraction
ls ~/datasets/freihand/training/rgb/ | head

# Should show: 00000000.jpg, 00000001.jpg, etc.
```

### HO-3D Issues

**OneDrive links not working**:
- Use Kaggle instead: https://www.kaggle.com/datasets/marcmarais/ho3d-v3
- OneDrive links expire/change, Kaggle is more reliable

**Missing depth maps**:
- HO-3D v3 has RGB + depth
- Some Kaggle versions may be RGB-only
- Use official OneDrive for complete dataset

### Kaggle Issues

**"401 Unauthorized"**:
```bash
# Check credentials
cat ~/.kaggle/kaggle.json
# Should show: {"username":"...","key":"..."}

# If missing, re-download from kaggle.com/settings
```

**"Dataset not found"**:
```bash
# Verify exact dataset name
kaggle datasets list -s freihand
kaggle datasets list -s ho3d
```

---

## Dataset Comparison

| Feature | FreiHAND | HO-3D |
|---------|----------|-------|
| **Purpose** | General hand pose | Hand-object interaction |
| **Images** | 134K | 103K |
| **Occlusion** | None | Severe (objects) |
| **Depth** | No | Yes (RGB-D) |
| **Objects** | No | Yes (YCB objects) |
| **Resolution** | 224×224 | 640×480 |
| **GT Accuracy** | ~5mm | ~5-10mm |
| **Difficulty** | Medium | Hard |
| **Download Size** | 4.4 GB | ~15-20 GB |

---

## References

### FreiHAND
- **Paper**: `docs/references/freihand2019.pdf` (in your repo)
- **Official**: https://lmb.informatik.uni-freiburg.de/projects/freihand/
- **GitHub**: https://github.com/lmb-freiburg/freihand
- **Kaggle**: https://www.kaggle.com/datasets/danieldelro/freihand

### HO-3D
- **Paper**: `docs/references/hOnnotate2020.pdf` (in your repo)
- **Official**: https://www.tugraz.at/index.php?id=40231
- **GitHub**: https://github.com/shreyashampali/ho3d
- **Kaggle**: https://www.kaggle.com/datasets/marcmarais/ho3d-v3

---

## Next Steps

After downloading:

1. **Test loaders**:
   ```bash
   python src/experiments/freihand_loader.py --num-samples 100
   ```

2. **Run Experiment 5**:
   ```bash
   python src/experiments/run_experiments.py --video test.mp4 --experiments 5
   ```

3. **Compare results**:
   - Your validation: 10.8mm
   - FreiHAND eval: ??? mm
   - HO-3D eval: ??? mm

4. **Generate paper figures**:
   ```bash
   python src/utils/visualize_experiments.py --results results/experiments
   ```
