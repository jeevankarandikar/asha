#!/bin/bash
#
# Download and setup FreiHAND and HO-3D datasets for Experiment 5
#
# Usage:
#   ./scripts/download_datasets.sh [freihand|ho3d|both]
#

set -e

DATASET_ROOT="${HOME}/datasets"
DATASET_TYPE="${1:-both}"

echo "========================================="
echo "Dataset Download Helper"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# FreiHAND Download
# ============================================================

download_freihand() {
    echo -e "${GREEN}[FreiHAND]${NC} Setting up FreiHAND dataset..."

    FREIHAND_DIR="${DATASET_ROOT}/freihand"
    mkdir -p "${FREIHAND_DIR}"
    cd "${FREIHAND_DIR}"

    echo "  → Directory: ${FREIHAND_DIR}"

    # Check if already downloaded
    if [ -d "training" ] && [ -f "training_xyz.json" ]; then
        echo -e "  ${YELLOW}⚠${NC}  FreiHAND already exists. Skipping download."
        return
    fi

    echo "  → Downloading FreiHAND training set v2 (3.7GB)..."

    # Download training data
    if ! wget --continue https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip; then
        echo -e "  ${RED}✗${NC} Download failed. Check your internet connection."
        exit 1
    fi

    echo "  → Extracting training set..."
    unzip -q FreiHAND_pub_v2.zip

    echo "  → Downloading evaluation set with annotations (724MB)..."
    wget --continue https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip

    echo "  → Extracting evaluation set..."
    unzip -q FreiHAND_pub_v2_eval.zip

    # Clean up zip files
    rm -f FreiHAND_pub_v2.zip FreiHAND_pub_v2_eval.zip

    echo -e "  ${GREEN}✓${NC} FreiHAND downloaded successfully!"
    echo "  → Location: ${FREIHAND_DIR}"
    echo ""
}

# ============================================================
# HO-3D Download
# ============================================================

download_ho3d() {
    echo -e "${GREEN}[HO-3D]${NC} Setting up HO-3D dataset..."

    HO3D_DIR="${DATASET_ROOT}/ho3d"
    mkdir -p "${HO3D_DIR}"

    echo "  → Directory: ${HO3D_DIR}"
    echo ""
    echo -e "${YELLOW}⚠${NC}  HO-3D has multiple download options:"
    echo ""
    echo "  ${GREEN}OPTION 1: Kaggle (Easiest)${NC}"
    echo "    1. Install Kaggle CLI: pip install kaggle"
    echo "    2. Setup API credentials: https://www.kaggle.com/docs/api"
    echo "    3. Download HO-3D v3:"
    echo "       kaggle datasets download -d marcmarais/ho3d-v3 -p ${HO3D_DIR}"
    echo "    4. Unzip: cd ${HO3D_DIR} && unzip ho3d-v3.zip"
    echo ""
    echo "  Or manually download from: https://www.kaggle.com/datasets/marcmarais/ho3d-v3"
    echo ""
    echo "  ${GREEN}OPTION 2: Official OneDrive (All versions)${NC}"
    echo "    HO-3D v2: https://1drv.ms/f/c/11742dd40d1cbdc1/ElPb2rhOCeRMg-dFSM3iwO8B5nS1SgnQJs9F6l28G0pKKg?e=TMuxgr"
    echo "    HO-3D v3: https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO"
    echo "    (Download manually via browser, extract to ${HO3D_DIR})"
    echo ""
    echo "  ${GREEN}OPTION 3: GitHub + Manual Links${NC}"
    echo "    git clone https://github.com/shreyashampali/ho3d.git ${HO3D_DIR}/ho3d_repo"
    echo "    # Follow instructions in README for dataset access"
    echo ""
}

# ============================================================
# Main
# ============================================================

echo "This script will download datasets to: ${DATASET_ROOT}"
echo ""

case ${DATASET_TYPE} in
    freihand)
        download_freihand
        ;;
    ho3d)
        download_ho3d
        ;;
    both)
        download_freihand
        download_ho3d
        ;;
    *)
        echo "Usage: $0 [freihand|ho3d|both]"
        exit 1
        ;;
esac

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Dataset locations:"
echo "  FreiHAND: ${DATASET_ROOT}/freihand"
echo "  HO-3D:    ${DATASET_ROOT}/ho3d"
echo ""
echo "Next steps:"
echo "  1. Update your code to point to these paths"
echo "  2. Run dataset loader to verify: python src/test_dataset_loader.py"
echo ""
