# v3: Transfer Learning (emg2pose Adaptation)

**Status**: ❌ ABANDONED (Fundamental Hardware Mismatch)

**Approach**: Adapt pre-trained emg2pose model (16ch @ 2kHz) to MindRove hardware (8ch @ 500Hz).

## Overview

v3 attempted **transfer learning** from Meta's emg2pose model (trained on 80M frames) to MindRove EMG hardware. After 2 weeks of design and implementation, the approach was **abandoned due to fundamental hardware incompatibility**.

**Key Finding**: Hardware mismatch (8ch→16ch spatial, 500Hz→2kHz temporal) cannot be bridged with simple adapters. Missing information cannot be synthesized.

## Architecture (Never Fully Trained)

**3-Stage Pipeline**:
```
MindRove EMG (8ch @ 500Hz)
    ↓
ChannelAdapter (8→16ch via learned expansion)
    ↓
FrequencyUpsampler (500Hz→2kHz via interpolation + learned refinement)
    ↓
Frozen emg2pose Model (TDS + LSTM → 21 joints)
    ↓
21 Hand Joint Positions
```

**Phase 1** (Planned):
- Train ChannelAdapter + FrequencyUpsampler
- Keep emg2pose frozen
- Target: Reconstruct spatial/temporal patterns from emg2pose's 16ch@2kHz data

**Phase 2** (Never Reached):
- Fine-tune last layers of emg2pose
- Adapt to MindRove signal characteristics

## Why It Failed

### Spatial Mismatch (8ch → 16ch)

**Problem**: emg2pose uses 16-channel custom EMG armband, MindRove has 8 channels.

**Attempted Solution**: ChannelAdapter to expand 8→16 channels
- Tried: Linear projection, learned expansion, interpolation
- **Issue**: Missing spatial information cannot be synthesized
- 8 channels cover fewer muscle groups → 16-channel patterns unrecoverable

**Fundamental Limit**: Cannot infer signals from muscles not measured

### Temporal Mismatch (500Hz → 2kHz)

**Problem**: emg2pose trained at 2kHz, MindRove samples at 500Hz (4x slower).

**Attempted Solution**: FrequencyUpsampler (500Hz→2kHz)
- Tried: Linear interpolation + learned refinement
- **Issue**: 4x upsampling creates artificial high-frequency content
- emg2pose learned temporal patterns specific to 2kHz signal dynamics

**Fundamental Limit**: True 2kHz signal has information missing from 500Hz (Nyquist limit at 250Hz)

### Model Mismatch

**emg2pose Architecture**:
- Input: [16, time] at 2kHz
- TDS blocks (Temporal Depth-wise Separable convolutions)
- LSTM with velocity prediction
- **Frozen weights** tuned for 16ch@2kHz statistics

**Incompatibility**: Adapter output never matched emg2pose's expected input distribution, even after extensive training attempts.

## What Was Implemented

**Files** (Archived):
- `train_mode1.py`: Phase 1 training (ChannelAdapter + FrequencyUpsampler)
- `train_mode2.py`: Phase 2 training (emg2pose fine-tuning)
- `record_transfer.py`: Data recording with dual output (joints + MANO θ)
- `inference_transfer.py`: Real-time inference (never functional)

**Models**:
- ChannelAdapter: 8→16 channel expansion
- FrequencyUpsampler: 500Hz→2kHz with learnable refinement
- Integration with frozen emg2pose checkpoint (236MB)

**Data**:
- Attempted to record training data for Phase 1
- Never collected sufficient data (abandoned before training phase)

## Timeline

- **Dec 13, 2024 (Morning)**: Full pipeline implemented (8 files, 2 training modes)
- **Dec 13, 2024 (Afternoon)**: Realized fundamental mismatch during debugging
- **Dec 13, 2024 (Evening)**: Decision to abandon transfer learning
- **Dec 13, 2024 (Night)**: Pivoted to direct training (v4/v5)
- **Dec 14, 2024**: v4 and v5 trained successfully with direct approach

**Effort**: 2 weeks planning + 1 day implementation → **Abandoned in <12 hours** after completion

## Why Direct Training (v4/v5) Succeeded

**v4/v5 Approach** (Direct Training):
- No hardware mismatch: Train on MindRove 8ch@500Hz directly
- Simpler architecture: Conv1D + LSTM (no TDS complexity)
- Small dataset: 20 min recording sufficient
- High-quality labels: v1 IK provides 9.71mm pseudo-labels

**Results**:
- v4 (EMG→θ): 34.92mm MPJPE
- v5 (EMG→joints): **11.95mm MPJPE** ✅

**Comparison**:
- v3 (transfer learning): Never trained, incompatible
- v5 (direct training): 11.95mm with 20 min data

## Key Lessons

1. **Hardware Compatibility Critical**: Cannot adapt models across different sensor configurations
2. **Missing Information Unfillable**: 8ch cannot synthesize 16ch, 500Hz cannot create 2kHz
3. **Over-Engineering Fails**: Complex 3-stage pipeline < simple direct training
4. **Pre-training Not Always Better**: 80M frames emg2pose < 20 min domain-specific data
5. **Simplicity Wins**: Direct Conv1D+LSTM outperforms complex transfer learning

## Full Postmortem

See `docs/TRANSFER_LEARNING_POSTMORTEM.md` for complete analysis of why transfer learning failed and lessons learned.

**Key Quote from Postmortem**:
> "The fundamental issue is that hardware compatibility matters more than model sophistication. You can't create information that wasn't measured in the first place."

## Research Contribution

v3's failure provides valuable negative results:

1. **Hardware mismatch is fundamental**: Cannot bridge 8ch↔16ch or 500Hz↔2kHz gaps
2. **Large pre-trained models unhelpful**: When domain mismatch is severe
3. **Direct training preferred**: For EMG tasks with custom hardware
4. **Small, domain-specific data > large pre-trained**: 20 min MindRove > 80M frames emg2pose

**Recommendation**: For EMG-to-hand-pose with custom hardware:
- ✅ Train models directly on your sensor configuration
- ✅ Use simple architectures (Conv1D + LSTM sufficient)
- ✅ Collect high-quality labels (IK pseudo-labels work well)
- ❌ Don't attempt transfer learning across hardware platforms
- ❌ Don't over-engineer (3-4 stage pipelines rarely help)

## Archived Files

**This directory contains historical reference only.** Do not use for active development.

- `train_mode1.py`: Phase 1 training script (adapter training)
- `train_mode2.py`: Phase 2 training script (fine-tuning)
- `record_transfer.py`: Data recording for transfer learning
- `inference_transfer.py`: Real-time inference (never functional)

For working EMG inference, use **v5** (`asha/v5/inference.py`).

## Timeline of Realization

**Dec 13, 2024, 9:00 AM**: "Let's adapt emg2pose with a 3-stage pipeline!"
**Dec 13, 2024, 6:00 PM**: "This doesn't work. Hardware mismatch is fundamental."
**Dec 13, 2024, 7:00 PM**: "Let's train from scratch instead."
**Dec 13, 2024, 11:00 PM**: v4 trained (34.92mm)
**Dec 14, 2024, 12:00 AM**: v5 trained (**11.95mm**) ✅

**Lesson**: Sometimes the "simpler" approach you dismissed at the start is actually the right answer.

---

**Status**: Permanently archived. Use v5 for EMG-to-hand-pose.
