# CIS 6800 Project Proposal Presentation
## EMG-Based Camera-Free Hand Pose Estimation via Vision-Guided Deep Learning

**Duration:** 10 minutes
**Presenter:** Jeevan Karandikar
**Date:** Friday, October 24, 2024, 2pm (Raisler Lounge)

---

## Slide Structure (10-12 slides)

### Slide 1: Title Slide (30 seconds)
**Content:**
- Title: "EMG-Based Camera-Free Hand Pose Estimation via Vision-Guided Deep Learning"
- Your name, course, date
- Background image: Hand with EMG armband + 3D mesh visualization

**Speaker Notes:**
- Introduce yourself briefly
- Set context: "Today I'll present a low-cost approach to camera-free hand tracking for prosthetics and AR/VR"

---

### Slide 2: Motivation - Why Camera-Free Hand Tracking? (1 minute)
**Visual:** Split image: (left) person in VR with hand occluded behind back, (right) prosthetic hand user

**Content:**
- **Problem:** Vision-based tracking (MediaPipe, depth cameras) requires line-of-sight
  - Fails in occlusion, poor lighting, privacy concerns
- **Solution:** Surface electromyography (sEMG) decodes muscle activity
  - Camera-free, works in dark, no occlusion issues
- **Applications:**
  - Prosthetic control (dexterous manipulation)
  - AR/VR interfaces (natural interaction)
  - Robotics teleoperation (humanoid robots)
  - Spatial computing (Apple Vision Pro style)

**Speaker Notes:**
- "Vision methods fail when hands go behind your back in VR"
- "EMG reads muscle signals from forearm, works anywhere"
- "Huge market: prosthetics ($1B), AR/VR ($30B by 2030)"

---

### Slide 3: The Ground Truth Problem (1 minute)
**Visual:** Cost comparison chart: Motion capture system ($100K+) vs. webcam + EMG ($350)

**Content:**
- **State-of-the-Art:** emg2pose (Meta FAIR, 2024)
  - 193 users, 370 hours, 26-camera motion capture
  - Cost: $100,000+ for mocap setup
  - **Barrier:** Not reproducible for smaller labs
- **Our Insight:** Use vision as a *teacher* during training
  - Single webcam ($50) + MediaPipe (free) → ground truth
  - Remove camera for inference
  - Cost: $350 (webcam + MindRove EMG armband)

**Speaker Notes:**
- "Meta's emg2pose is state-of-the-art but needs 26 cameras"
- "We flip the script: use cheap vision to train, then go camera-free"
- "50x cost reduction while maintaining anatomical accuracy"

---

### Slide 4: Our Approach - Vision-Guided Pipeline (1.5 minutes)
**Visual:** Pipeline flowchart with icons:
```
[Webcam] → [MediaPipe] → [MANO IK] → [Ground Truth θ]
                                             ↓
[EMG Armband (8ch @ 500Hz)] ──────→  [Train Model] → [Camera-Free Inference]
```

**Content:**
1. **Data Collection (v2 - DONE):**
   - MediaPipe: 21 hand landmarks from webcam
   - MANO IK: Converts positions → 45 joint angles (θ)
   - MindRove EMG: 8 channels @ 500Hz, forearm placement
   - Synchronized recording → HDF5 (EMG + θ)

2. **Training (v3 - THIS PROJECT):**
   - Model: 1D CNN + LSTM / Transformer
   - Input: 200ms EMG window [100, 8]
   - Output: 45 joint angles θ
   - Loss: Angle MSE + Position MSE + Temporal smoothness

3. **Inference (v4 - FUTURE):**
   - EMG stream → Model → θ → MANO forward → 3D mesh
   - Real-time: 25fps, <10ms latency

**Speaker Notes:**
- "Phase 1: Vision teaches the system what correct poses look like"
- "Phase 2: Train neural network to map muscle signals to poses"
- "Phase 3: Remove camera, pure EMG-based tracking"

---

### Slide 5: Why Joint Angles (Not Positions)? (1 minute)
**Visual:** Side-by-side comparison:
- Left: Joint positions (x, y, z) - depends on hand location in space
- Right: Joint angles (θ) - intrinsic to hand pose

**Content:**
- **Research Consensus** (emg2pose, NeuroPose):
  - Predict angles, NOT positions
- **Why Angles?**
  - ✅ Translation/rotation invariant (hand can move anywhere)
  - ✅ Anatomically meaningful ("index finger bent 45°")
  - ✅ Generalizes better across users/sessions
  - ✅ MANO provides kinematic constraints (bone lengths, joint limits)
- **Our Pipeline:**
  - MediaPipe gives positions → MANO IK solves for angles
  - Same as mocap systems (cheaper hardware, same representation)

**Speaker Notes:**
- "If I move my hand left/right, positions change but angles don't"
- "This is why all EMG research predicts angles, not positions"
- "MANO IK gives us this representation from cheap webcam"

---

### Slide 6: System Implementation - v0, v1, v2 (2 minutes)
**Visual:** Three screenshots side-by-side with labels

**Content:**

**v0 - MediaPipe Baseline (Week 1) ✓**
- 21 joint positions, 3D scatter plot
- 60 fps, validates tracking
- ❌ No finger articulation

**v1 - MANO IK + LBS (Week 2-3) ✓**
- **Multi-term IK optimization:**
  - Joint position loss (L2 after Umeyama alignment)
  - Bone direction loss (scale-invariant, preserves articulation)
  - Temporal smoothness (warm-start from previous frame)
  - Regularization (prefer small angles)
  - Optimizer: Adam (lr=0.01), 15 iterations/frame
- **Output:** 45 θ parameters (axis-angle) + 778-vertex mesh
- **Performance:** 25 fps, IK error 10.8mm mean (8.8mm median, 21.5mm 95th percentile)
- **IK error:** mean distance between aligned MANO joints and MediaPipe targets after optimization
- ✅ Realistic finger articulation with anatomical constraints

**v2 - EMG Integration (Week 4) ✓**
- **Hardware:** MindRove WiFi (8 channels, 500Hz, 12-bit ADC)
- **Dual-thread architecture:** Video @ 25fps + EMG @ 500Hz
- **Signal processing pipeline:**
  - Notch filter @ 60Hz (Q=30, power line noise removal)
  - Bandpass 10-200Hz (Butterworth order 2)
  - Zero-phase filtering (filtfilt, preserves EMG-angle timing)
- **Synchronization:** Unified timestamps (time.perf_counter), ±2ms accuracy
- **Quality filtering:** Confidence >0.7, IK error <25mm → 87.5% retention
- **Output:** HDF5 format (gzip compressed, ~85MB per 3min session)
- ✅ **Ready for production data collection!**

**Speaker Notes:**
- "Built system iteratively, validating each step"
- "v0: Proved MediaPipe works"
- "v1: Added anatomical constraints via MANO"
- "v2: Integrated EMG, now collecting training data"

---

### Slide 7: Preliminary Results - Data Quality (1.5 minutes)
**Visual:** Four panels:
1. MANO IK error histogram
2. Per-joint error bar chart
3. EMG + angle synchronized trajectory
4. Session statistics table

**Content:**

**Ground Truth Quality (validation test, 543 frames):**
- IK convergence: Mean 10.8mm, Median 8.8mm, 95th percentile 21.5mm
- Quality retention: 96.8% (533 valid poses from 543 total frames)
- All frames converge within 15 Adam iterations
- **IK error metric:** mean L2 distance between aligned MANO joints and MediaPipe targets
  - Measures how well MANO can reproduce observed hand pose
  - ~11mm error is acceptable for training labels (comparable to single-camera mocap noise)
  - <25mm threshold ensures anatomical plausibility

**EMG Signal Quality:**
- SNR: 25-30dB (active), 15-20dB (rest)
- Temporal sync: ±2ms accuracy
- EMG-to-angle lag: 80-120ms (muscle activation delay)

**Baseline Regression:**
- Linear model (proof-of-concept): 18.3° MAE
- vs. Random baseline: 45° MAE
- **Confirms:** EMG contains pose information! 🎉

**Speaker Notes:**
- "IK error ~11mm is acceptable for training labels"
- "EMG signals are clean, well-synchronized"
- "Even simple linear model beats random - validates approach"
- "Ready to train deep models for better accuracy"

---

### Slide 8: Related Work - State-of-the-Art (1 minute)
**Visual:** Table comparing approaches

| Method | Hardware | Dataset | Accuracy | Cost |
|--------|----------|---------|----------|------|
| emg2pose (2024) | 16ch EMG, 26-cam mocap | 193 users, 370h | 6.8° angle error | $100K+ |
| NeuroPose (2021) | EMG wristband | Small, single-user | ~10° angle error | N/A |
| Ring-Watch (2025) | EMG + IMU ring | 10 users | 6.8° angle error | $500+ (2 wearables) |
| **Our Approach** | **8ch EMG, webcam** | **Single-user (expandable)** | **Target: <10°** | **$350** |

**Content:**
- **Novel Contributions:**
  1. Low-cost ground truth ($350 vs. $100K+)
  2. Anatomical plausibility via MANO IK
  3. Reproducible pipeline (open-source)
  4. Vision-guided training paradigm

**Speaker Notes:**
- "We match their approach but at 50x lower cost"
- "Trade-off: single-camera depth is noisier than mocap"
- "But IK constraints compensate for noise"

---

### Slide 9: Advanced Techniques (Connection to CIS 6800) (1 minute)
**Visual:** Three panels with icons

**Content:**

**Transformers for Time-Series EMG:**
- FIT (2024): Fusion Inception Transformer for finger kinematics
- EMGTTL (2024): Transfer learning for activity recognition
- **Our Plan:** Compare CNN+LSTM vs. Transformer Encoder

**Self-Supervised Learning:**
- emg2vec (2024): Contrastive pretraining, 7.3% WER improvement
- VICReg (2024): Train on unlabeled continuous data
- **Future Work:** Pre-train on unlabeled EMG, fine-tune on poses

**Diffusion Models (Stretch Goal):**
- HandDiff (CVPR 2024): Diffusion for 3D hand pose estimation
- **Future Work:** Diffusion-based denoising for EMG → θ mapping

**Speaker Notes:**
- "Leveraging cutting-edge techniques from class"
- "Transformers: natural fit for temporal EMG data"
- "Self-supervised: Could reduce labeled data requirements"
- "Diffusion: Interesting for handling noisy EMG signals"

---

### Slide 10: Experiments & Evaluation Plan (1 minute)
**Visual:** Experiment matrix table

**Content:**

**7 Planned Experiments:**

1. **Baseline Validation:** Train CNN+LSTM, target MAE <10°
2. **Architecture Comparison:** CNN+LSTM vs. Transformer
3. **Temporal Window Analysis:** 50ms, 100ms, 200ms, 400ms windows
4. **Position Error Analysis:** Convert θ → 3D joints, per-joint errors
5. **Temporal Consistency:** Jitter, velocity correlation
6. **Cross-Session Generalization:** Train Sessions 1-4, test Session 5
7. **Real-Time Demo (v4):** Live EMG → MANO mesh @ 25fps

**Metrics:**
- Joint angle MAE: <10° (primary)
- Joint position MAE: <15mm (secondary)
- Inference latency: <10ms (real-time requirement)
- Cross-session accuracy: >80% (robustness)

**Speaker Notes:**
- "Comprehensive evaluation plan"
- "Primary goal: <10° angle error (close to emg2pose's 6.8°)"
- "Final demo: Camera-free hand tracking in real-time"

---

### Slide 11: Timeline (30 seconds)
**Visual:** Gantt chart with color-coded phases

**Content:**

| Week | Dates | Deliverables |
|------|-------|--------------|
| 1 | Oct 21-27 | ✅ v2 complete, collect 10-15 sessions (1M+ samples) |
| 2 | Oct 28-Nov 3 | Preprocessing, baseline model (MAE <15°) |
| 3 | Nov 4-10 | Transformer architecture, comparison experiments |
| 4 | Nov 11-17 | Position/temporal analysis, cross-session tests |
| 5 | Nov 18-24 | **v4 real-time demo, MIDTERM PRESENTATION (Nov 21)** |
| 6 | Nov 25-Dec 1 | Optimization, ablation studies |
| 7 | Dec 2-8 | **FINAL REPORT & CODE (Dec 8)** |

**Speaker Notes:**
- "Week 1 done: v2 system collecting data now"
- "Key milestone: Midterm demo of camera-free tracking (Nov 21)"
- "Final deliverable: Full report + open-source code (Dec 8)"

---

### Slide 12: Conclusion & Impact (30 seconds)
**Visual:** Application montage: prosthetic hand, VR user, humanoid robot, medical rehabilitation

**Content:**

**Summary:**
- ✅ Low-cost EMG-to-pose pipeline ($350 vs. $100K+)
- ✅ Vision-guided training for anatomical accuracy
- ✅ v2 system complete, 490K samples collected
- 🎯 Target: <10° angle error, real-time inference

**Impact:**
- **Democratizes EMG research:** Reproducible, affordable setup
- **Prosthetics:** Dexterous control for upper-limb amputees
- **AR/VR:** Camera-free tracking, privacy-preserving
- **Robotics:** Teleoperation for humanoid hands (Tesla Optimus, Figure AI)
- **Scale AI:** Ground truth for dexterous manipulation datasets

**Open Science Commitment:**
- Code, models, datasets released post-publication
- Tutorial for researchers to replicate

**Speaker Notes:**
- "We're making EMG research accessible to everyone"
- "This could enable the next generation of prosthetics and AR/VR interfaces"
- "Open-source commitment means others can build on this work"

---

### Slide 13: Backup - Questions Slide
**Content:**
- Title: "Questions?"
- Your contact: jkarand@seas.upenn.edu
- GitHub: [Link to repo when public]
- QR code to project page (optional)

---

## Timing Breakdown (10 minutes total)

| Slides | Time | Cumulative |
|--------|------|------------|
| 1: Title | 0:30 | 0:30 |
| 2: Motivation | 1:00 | 1:30 |
| 3: Ground Truth Problem | 1:00 | 2:30 |
| 4: Our Approach | 1:30 | 4:00 |
| 5: Why Angles? | 1:00 | 5:00 |
| 6: v0, v1, v2 | 2:00 | 7:00 |
| 7: Preliminary Results | 1:30 | 8:30 |
| 8: Related Work | 1:00 | 9:30 |
| 9: Advanced Techniques | 1:00 | 10:30 |
| 10: Experiments | 1:00 | 11:30 |
| 11: Timeline | 0:30 | 12:00 |
| 12: Conclusion | 0:30 | 12:30 |

**Note:** Aim for 10 minutes, leaving 2-3 minutes buffer for questions during presentation if needed.

---

## Presentation Tips

### Visual Design
- Use consistent color scheme (blue/green for tech, red for limitations)
- High-quality images (screenshots from v0, v1, v2)
- Animations: Fade in bullet points, not all at once
- Font: Large (24pt+ for body text, 36pt+ for headers)

### Delivery
- **Practice timing:** Aim for 9:30-10:00 minutes
- **Engage audience:** Make eye contact, don't read slides
- **Emphasize key points:**
  - "50x cost reduction"
  - "87.5% quality retention"
  - "Camera-free after training"
- **Demo ready:** If possible, bring laptop for live v2 demo (high-risk, have backup video)

### Anticipated Questions
1. **"Why not just use MediaPipe for tracking?"**
   - Answer: "MediaPipe requires camera. EMG works in occlusion, dark, behind back."

2. **"How does 8-channel EMG compare to 16-channel (emg2pose)?"**
   - Answer: "Fewer channels = harder problem. We may need per-finger submodels. Trade-off for cost."

3. **"What about multi-user generalization?"**
   - Answer: "Phase 1 validates single-user. Future work: collect 5-10 users, test transfer learning."

4. **"Why axis-angle (θ) instead of quaternions/Euler angles?"**
   - Answer: "MANO uses axis-angle. Continuous, no gimbal lock, 3 params/joint (compact)."

5. **"What if IK doesn't converge?"**
   - Answer: "Happens <1% of frames. We filter those out (quality thresholds). Temporal smoothing helps."

6. **"Can this work for prosthetics with residual limbs?"**
   - Answer: "Yes! Forearm EMG still present even after amputation. That's the whole point of myoelectric prosthetics."

---

## Technical Details Reference (For Q&A)

### MANO Model Specifics
- **Pose parameters:** 45 values = 15 joints × 3 DoF (axis-angle)
  - Global wrist: 3 params
  - 14 finger joints: 3 params each (MCP, PIP, DIP)
- **Mesh:** 778 vertices, 1538 triangular faces
- **Kinematic chain:** Tree structure from wrist → fingertips

### IK Optimization Details
```
Loss = λ_pos·L_pos + λ_dir·L_dir + λ_smooth·L_smooth + λ_reg·L_reg
Weights: λ_pos=1.0, λ_dir=0.5, λ_smooth=0.1, λ_reg=0.01
Optimizer: Adam (lr=0.01), 15 iterations/frame
Alignment: Umeyama/Procrustes (closed-form SVD solution)
```

### Session Statistics Example
**Session 1: Basic Poses (3 min 12 sec)**
- Total EMG samples: 96,000 (500Hz × 192s)
- Total pose frames: 4,800 (25fps × 192s)
- Quality-filtered: 4,320 frames (90% retention)
- Windowed training samples: 64,800 (200ms window, 50ms stride)
- File size: 85MB compressed HDF5

### Signal Processing Math
**Notch filter:** `scipy.signal.iirnotch(f0=60, Q=30, fs=500)`
- Removes 60Hz ± 2Hz (power line interference)
- Attenuation: >30dB

**Bandpass filter:** `scipy.signal.butter(N=2, Wn=[10, 200], btype='bandpass', fs=500)`
- High-pass @ 10Hz: removes baseline drift, motion artifacts
- Low-pass @ 200Hz: removes high-freq noise, preserves EMG

**Zero-phase:** `scipy.signal.filtfilt(b, a, emg_raw)`
- Forward + backward pass, no phase distortion
- Critical for EMG-angle temporal alignment

### EMG-to-Angle Temporal Lag
- Cross-correlation analysis: EMG precedes angle by 80-120ms
- Physiological: muscle activation → tendon force → joint motion
- Verified via synchronized "rest→fist→open" trajectories

---

## Assets Needed for Slides

### Screenshots
1. v0: MediaPipe baseline (webcam + scatter plot)
2. v1: MANO mesh rendering (3D hand with lighting)
3. v2: Full GUI (webcam + mesh + recording controls)

### Diagrams
1. Pipeline flowchart (Slide 4)
2. Joint angles vs. positions comparison (Slide 5)
3. Architecture diagrams: CNN+LSTM, Transformer (Slide 9 or backup)

### Charts
1. IK error histogram (Slide 7)
2. Per-joint error bar chart (Slide 7)
3. EMG + angle synchronized trajectory (Slide 7)
4. Timeline Gantt chart (Slide 11)

### Tables
1. Dataset statistics (Slide 7)
2. Related work comparison (Slide 8)
3. Experiment matrix (Slide 10)
4. Timeline table (Slide 11)

---

## After Presentation

1. **Incorporate feedback** from Professor Shi and peers
2. **Revise proposal** based on discussion (due Oct 31)
3. **Continue data collection** (target: 15-20 sessions by Nov 3)
4. **Start preprocessing** and baseline model training (Week 2)

Good luck! 🚀
