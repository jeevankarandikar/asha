# CIS 6800 Final Project - Presentation Session Handoff

**Context**: I just finished polishing my final report (CIS6800_Final.tex). Now I need to prepare a 10-15 minute presentation and record a demo video for my final project defense.

## What I Need Help With

### 1. **Presentation Creation**
- **Goal**: 10-15 minute presentation (~12-15 slides)
- **Audience**: CIS 6800 class (computer vision, ML background)
- **Key message**: IK pseudo-labeling enables camera-free EMG hand tracking without $100K+ mocap
- **Tone**: Same as the report - direct, honest about failures, focused on practical results

**Presentation Structure** (suggested):
1. **Problem** (1 slide): Why EMG? Camera limitations, mocap cost barrier
2. **Our Approach** (1 slide): IK pseudo-labeling → EMG training
3. **System Overview** (1 slide): Two-stage pipeline (IK → EMG)
4. **IK Validation** (3 slides):
   - Exp 1: Loss ablation (bone direction critical)
   - Exp 2-3: Alignment + optimizer
   - Exp 4: Per-joint error
5. **EMG Training** (3 slides):
   - v4 vs v5 architecture comparison
   - Training convergence
   - Ground truth quality analysis
6. **Results** (2 slides):
   - Version evolution timeline
   - SOTA comparison (14.92mm vs HandFormer 12.33mm)
7. **What Failed** (1 slide): Image training (47mm), transfer learning (hardware mismatch)
8. **Demo Video** (2-3 min): mano.py showing IK pseudo-labeling in action
9. **Conclusion** (1 slide): Camera-free @ 14.92mm, accessible research

**Assets Available**:
- ✅ Final report PDF: `/Users/jeevankarandikar/Downloads/CIS_6800_Final_Project_Report.pdf`
- ✅ 8 figures in `docs/figures/`:
  - exp1_loss_ablation.png
  - exp2_alignment.png
  - exp3_optimizer.png
  - exp4_per_joint.png
  - comparison_v4_vs_v5.png
  - v5_training.png
  - version_timeline.png
  - comparison_state_of_art.png
- ✅ All 7 experiments documented in LaTeX
- ✅ Full codebase in `src/` (MANO IK, EMG training, inference)

### 2. **Demo Video Planning**
**Recommendation**: Record 2-3 minute video of `mano.py` (MANO IK pseudo-labeling)

**Why mano.py?**
- Shows the core innovation (9.71mm IK pseudo-labels that enable everything)
- Most polished visually (3D hand mesh rendering)
- No EMG hardware needed (just laptop webcam)
- Real-time at 25fps with smooth tracking
- Demonstrates what makes the whole approach work

**Demo Script** (2-3 minutes):
1. **Introduction** (15 sec): "This is our IK pseudo-labeling system generating training data for EMG models"
2. **Show Hand Movements** (60 sec):
   - Open hand → fist
   - Individual finger movements
   - Pinch, point gestures
   - Show real-time mesh tracking
3. **Highlight Quality** (30 sec): "This achieves 9.71mm accuracy, better than HandFormer's 10.92mm stereo baseline"
4. **Explain Value** (15 sec): "These pseudo-labels trained our EMG model to 14.92mm - no motion capture needed"
5. **Optional**: Split-screen showing webcam + 3D mesh side by side

**How to Record**:
```bash
# Terminal
cd /Users/jeevankarandikar/Documents/GitHub/asha
source asha_env/bin/activate
python src/programs/mano.py

# Use QuickTime Player or OBS Studio to record screen
# Crop to just the GUI window
# Export as MP4 (1920x1080 @ 30fps recommended)
```

### 3. **Tasks for This Session**

1. **Read the final report PDF** to understand the complete story
2. **Create presentation outline** with slide-by-slide breakdown
3. **Draft speaker notes** for each slide (what to say, key points to emphasize)
4. **Refine demo video script** with exact timing and movements
5. **Suggest presentation platform**: Google Slides? PowerPoint? Keynote?
6. **Answer questions** about technical details, results interpretation, or presentation strategy

## Key Results to Emphasize

**Numbers that matter**:
- **9.71mm IK** (our pseudo-labels) vs **10.92-12.33mm HandFormer** (SOTA vision)
- **14.92mm EMG v5** (camera-free) vs **34.92mm EMG v4** (2.3× improvement)
- **20 minutes training data** vs **80M frames emg2pose** (400,000× less data)
- **~$600 EMG armband** vs **$100K+ mocap** (167× cheaper for deployment)
- **30fps real-time** (inference), **25fps** (IK pseudo-labeling)

**What makes this work**:
1. Multi-term IK loss (position + bone direction + temporal smoothness)
2. Direct joint prediction (v5) beats parametric θ (v4) by 2.3×
3. High-quality pseudo-labels (9.71mm) enable strong EMG models
4. Simple architecture (Conv1D + LSTM) sufficient - no over-engineering

**Honest failures** (important to include):
- Image→pose training plateaued at 47mm (all models: ResNet, Transformer)
- Transfer learning from emg2pose failed (8ch vs 16ch hardware mismatch)
- These failures validated the direct approach

## Technical Details You Might Need

**IK Optimization**:
- Adam optimizer, 15 iterations/frame, lr=0.01
- Loss weights: λ_pos=1.0, λ_dir=0.5, λ_smooth=0.1, λ_reg=0.01
- Umeyama alignment with scale estimation
- 99.7% convergence rate (5,330 frames)

**EMG Training**:
- Model: Conv1D (8→64→128) + 2-layer LSTM (256 hidden)
- Data: 5 sessions × 4 min = 20 min (23,961 windows @ 500Hz)
- Training: A100 GPU, 100 epochs, ~1.5 hours
- Final: Val MPJPE 14.92mm (epoch 99), LR schedule 1e-3 → 5e-4 @ epoch 78

**Validation**:
- 7 experiments total (4 IK validation + 3 EMG training)
- Single-subject data (limitation acknowledged)
- Systematic ablation studies (loss terms, alignment, optimizer)

## Questions I Might Have

1. Should I show both v4 and v5 results or focus on final v5?
2. How much time to spend on failed approaches (image training, transfer learning)?
3. Should I include live demo or just recorded video?
4. How to frame publishability (workshop vs conference)?
5. What Q&A questions should I prepare for?

## Files You Can Access

**Report**:
- `/Users/jeevankarandikar/Downloads/CIS_6800_Final_Project_Report.pdf`
- `/Users/jeevankarandikar/Documents/GitHub/asha/docs/reports/CIS6800_Final.tex`

**Figures**:
- `/Users/jeevankarandikar/Documents/GitHub/asha/docs/figures/*.png` (all 8 figures)

**Code** (for understanding):
- `/Users/jeevankarandikar/Documents/GitHub/asha/src/programs/mano.py` (IK demo)
- `/Users/jeevankarandikar/Documents/GitHub/asha/src/model_utils/pose_fitter.py` (IK implementation)
- `/Users/jeevankarandikar/Documents/GitHub/asha/src/training/train_joints_colab.py` (v5 training)

**Documentation**:
- `/Users/jeevankarandikar/Documents/GitHub/asha/CLAUDE.md` (full project history)
- `/Users/jeevankarandikar/Documents/GitHub/asha/docs/TRANSFER_LEARNING_POSTMORTEM.md` (why transfer learning failed)

---

**Deliverable**: By end of this session, I should have:
1. ✅ Complete presentation outline (12-15 slides)
2. ✅ Speaker notes for each slide
3. ✅ Refined demo video script with timing
4. ✅ Answers to any technical questions for Q&A prep

Let's make this presentation as direct and compelling as the report!
