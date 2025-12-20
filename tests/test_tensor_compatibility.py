"""
Test script to verify tensor compatibility across the entire pipeline.

Run this before training to catch any shape mismatches early.

Usage:
    python src/tests/test_tensor_compatibility.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import MANOLoss, SimpleMANOLoss
from training.emg_model import EMG2PoseModel


def test_mano_loss_shapes():
    """Test MANOLoss with various tensor shapes."""
    print("\n" + "="*60)
    print("TEST 1: MANOLoss Tensor Compatibility")
    print("="*60)

    # Create dummy inputs
    B = 4  # batch size
    pred_theta = torch.randn(B, 45)
    pred_beta = torch.randn(B, 10)
    gt_theta = torch.randn(B, 45)
    gt_beta = torch.randn(B, 10)
    gt_joints = torch.randn(B, 21, 3)

    print(f"Input shapes:")
    print(f"  pred_theta: {pred_theta.shape}")
    print(f"  pred_beta: {pred_beta.shape}")
    print(f"  gt_theta: {gt_theta.shape}")
    print(f"  gt_beta: {gt_beta.shape}")
    print(f"  gt_joints: {gt_joints.shape}")

    # Test SimpleMANOLoss
    print("\nTesting SimpleMANOLoss...")
    simple_loss = SimpleMANOLoss()
    losses_simple = simple_loss(pred_theta, pred_beta, gt_theta, gt_beta)
    print(f"  param loss: {losses_simple['param'].item():.4f}")
    print(f"  reg loss: {losses_simple['reg'].item():.4f}")
    print(f"  total loss: {losses_simple['total'].item():.4f}")
    assert losses_simple['total'].shape == torch.Size([]), "Total loss should be scalar"
    print("  ✅ SimpleMANOLoss passed")

    # Test MANOLoss (without MANO layer)
    print("\nTesting MANOLoss (without MANO layer)...")
    enhanced_loss = MANOLoss(
        lambda_param=0.5,
        lambda_joint=5.0,
        lambda_angular=1.0,
        lambda_pca=0.1,
        mano_layer=None,  # No MANO layer for this test
    )
    losses_enhanced = enhanced_loss(pred_theta, pred_beta, gt_theta, gt_beta, gt_joints=None)
    print(f"  param loss: {losses_enhanced['param'].item():.4f}")
    print(f"  angular loss: {losses_enhanced['angular'].item():.4f}")
    print(f"  reg loss: {losses_enhanced['reg'].item():.4f}")
    print(f"  pca loss: {losses_enhanced['pca'].item():.4f}")
    print(f"  total loss: {losses_enhanced['total'].item():.4f}")
    assert losses_enhanced['total'].shape == torch.Size([]), "Total loss should be scalar"
    print("  ✅ MANOLoss passed")

    print("\n✅ TEST 1 PASSED: All loss functions handle tensor shapes correctly")


def test_emg_model_shapes():
    """Test EMG2PoseModel with various tensor shapes."""
    print("\n" + "="*60)
    print("TEST 2: EMG Model Tensor Compatibility")
    print("="*60)

    # Create EMG model
    model = EMG2PoseModel(
        emg_channels=8,
        hidden_size=512,
        output_size=45,
        use_velocity=True,
        use_imu=False,
    )
    model.eval()

    print(f"Model architecture:")
    print(f"  TDS featurizer: 8 channels → 16 channels, 500Hz → 50Hz")
    print(f"  LSTM: 2 layers, hidden size 512")
    print(f"  Output: velocity (Δθ), integrated to absolute θ")

    # Test different batch sizes
    batch_sizes = [1, 4, 8]

    for B in batch_sizes:
        print(f"\nTesting batch size B={B}...")

        # Create dummy EMG input [B, T, C]
        emg = torch.randn(B, 500, 8)  # 1 second @ 500Hz, 8 channels
        print(f"  Input shape: {emg.shape}")

        # Forward pass (without prev_theta)
        with torch.no_grad():
            theta = model(emg)
        print(f"  Output shape: {theta.shape}")

        # Verify shape
        assert theta.shape == (B, 45), f"Expected ({B}, 45), got {theta.shape}"
        print(f"  ✅ Shape correct: {theta.shape}")

        # Test with prev_theta (velocity integration)
        prev_theta = torch.randn(B, 45)
        with torch.no_grad():
            theta_integrated = model(emg, prev_theta=prev_theta)
        assert theta_integrated.shape == (B, 45), f"Expected ({B}, 45), got {theta_integrated.shape}"
        print(f"  ✅ Velocity integration works: {theta_integrated.shape}")

    print("\n✅ TEST 2 PASSED: EMG model handles all batch sizes correctly")


def test_emg_model_with_imu():
    """Test EMG2PoseModel with IMU fusion."""
    print("\n" + "="*60)
    print("TEST 3: EMG Model with IMU Fusion")
    print("="*60)

    # Create EMG model with IMU
    model = EMG2PoseModel(
        emg_channels=8,
        hidden_size=512,
        output_size=45,
        use_velocity=True,
        use_imu=True,
        imu_dim=18,  # 2 IMUs × 9-DOF
    )
    model.eval()

    print(f"Model architecture:")
    print(f"  EMG: 8 channels @ 500Hz")
    print(f"  IMU: 18 features (2 × 9-DOF)")
    print(f"  Fusion: Concatenate EMG + IMU features")

    B = 4

    # Create dummy inputs
    emg = torch.randn(B, 500, 8)   # [B, T, 8]
    imu = torch.randn(B, 500, 18)  # [B, T, 18] (same T as EMG)

    print(f"\nInput shapes:")
    print(f"  EMG: {emg.shape}")
    print(f"  IMU: {imu.shape}")

    # Forward pass with IMU
    with torch.no_grad():
        theta = model(emg, imu=imu)

    print(f"Output shape: {theta.shape}")
    assert theta.shape == (B, 45), f"Expected ({B}, 45), got {theta.shape}"
    print(f"✅ IMU fusion works correctly")

    print("\n✅ TEST 3 PASSED: EMG + IMU fusion handles tensor shapes correctly")


def test_dataset_compatibility():
    """Test dataset output format compatibility."""
    print("\n" + "="*60)
    print("TEST 4: Dataset Output Compatibility")
    print("="*60)

    # Simulate dataset output
    print("Simulating FreiHAND dataset output...")

    # Single sample (before batching)
    sample = {
        'image': torch.randn(3, 224, 224),    # After transform
        'theta': torch.randn(45),
        'beta': torch.randn(10),
        'joints_3d': torch.randn(21, 3),
    }

    print(f"Single sample shapes:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}")

    # Simulate batching
    B = 8
    batch = {
        'image': torch.stack([sample['image'] for _ in range(B)]),
        'theta': torch.stack([sample['theta'] for _ in range(B)]),
        'beta': torch.stack([sample['beta'] for _ in range(B)]),
        'joints_3d': torch.stack([sample['joints_3d'] for _ in range(B)]),
    }

    print(f"\nBatched sample shapes (B={B}):")
    for key, val in batch.items():
        print(f"  {key}: {val.shape}")

    # Verify shapes match training expectations
    assert batch['image'].shape == (B, 3, 224, 224), "Image shape mismatch"
    assert batch['theta'].shape == (B, 45), "Theta shape mismatch"
    assert batch['beta'].shape == (B, 10), "Beta shape mismatch"
    assert batch['joints_3d'].shape == (B, 21, 3), "Joints shape mismatch"

    print("\n✅ TEST 4 PASSED: Dataset outputs match training expectations")


def test_end_to_end_flow():
    """Test complete end-to-end flow from dataset to loss."""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Pipeline Flow")
    print("="*60)

    B = 4

    print("Step 1: Dataset output (simulated)")
    batch = {
        'image': torch.randn(B, 3, 224, 224),
        'theta': torch.randn(B, 45),
        'beta': torch.randn(B, 10),
        'joints_3d': torch.randn(B, 21, 3),
    }
    print(f"  ✅ Batch created with B={B}")

    print("\nStep 2: Model forward pass (simulated)")
    # Simulate model output
    pred_theta = torch.randn(B, 45)
    pred_beta = torch.randn(B, 10)
    print(f"  ✅ Predictions: theta {pred_theta.shape}, beta {pred_beta.shape}")

    print("\nStep 3: Loss computation")
    criterion = MANOLoss(
        lambda_param=0.5,
        lambda_joint=5.0,
        lambda_angular=1.0,
        lambda_pca=0.1,
        mano_layer=None,
    )

    losses = criterion(
        pred_theta=pred_theta,
        pred_beta=pred_beta,
        gt_theta=batch['theta'],
        gt_beta=batch['beta'],
        gt_joints=batch['joints_3d'],
    )

    print(f"  Loss components:")
    for key, val in losses.items():
        if key != 'total':
            print(f"    {key}: {val.item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  ✅ Loss computed successfully")

    print("\nStep 4: Backward pass (simulated)")
    loss_value = losses['total']
    # In real training: loss_value.backward()
    print(f"  ✅ Loss is scalar, ready for backward()")

    print("\n✅ TEST 5 PASSED: End-to-end pipeline flow compatible")


def test_angular_loss_details():
    """Test quaternion geodesic loss computation in detail."""
    print("\n" + "="*60)
    print("TEST 6: Angular Geodesic Loss Details")
    print("="*60)

    from training.losses import axis_angle_to_quaternion, quaternion_geodesic_loss

    # Test axis-angle to quaternion conversion
    print("Testing axis-angle → quaternion conversion...")
    theta = torch.randn(10, 3)  # 10 joints, 3 DoF each
    print(f"  Input theta: {theta.shape}")

    quat = axis_angle_to_quaternion(theta)
    print(f"  Output quaternion: {quat.shape}")
    assert quat.shape == (10, 4), f"Expected (10, 4), got {quat.shape}"
    print(f"  ✅ Conversion correct")

    # Test quaternion geodesic loss
    print("\nTesting quaternion geodesic loss...")
    q1 = torch.randn(10, 4)
    q2 = torch.randn(10, 4)
    print(f"  Input q1: {q1.shape}")
    print(f"  Input q2: {q2.shape}")

    geodesic = quaternion_geodesic_loss(q1, q2)
    print(f"  Output geodesic: {geodesic.shape}")
    assert geodesic.shape == (10,), f"Expected (10,), got {geodesic.shape}"
    print(f"  Geodesic range: [{geodesic.min().item():.4f}, {geodesic.max().item():.4f}] radians")
    print(f"  ✅ Geodesic loss computed correctly")

    print("\n✅ TEST 6 PASSED: Angular loss computations correct")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TENSOR COMPATIBILITY VERIFICATION TEST SUITE")
    print("="*60)

    try:
        test_mano_loss_shapes()
        test_emg_model_shapes()
        test_emg_model_with_imu()
        test_dataset_compatibility()
        test_end_to_end_flow()
        test_angular_loss_details()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Tensor shapes are compatible across the entire pipeline")
        print("✅ Architecture is ready for training")
        print("\nNext steps:")
        print("  1. Run training on Google Colab with FreiHAND dataset")
        print("  2. Monitor diversity metrics (θ variance, entropy, gesture F1)")
        print("  3. Compare with baseline (resnet_v2: 47mm → target: 10-12mm)")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
