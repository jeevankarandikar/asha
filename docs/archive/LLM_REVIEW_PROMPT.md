# LLM Review Prompt for Project Asha

Copy this prompt and paste it into Grok, ChatGPT, Gemini, or another LLM along with the contents of `CLAUDE.md`.

---

## Prompt

You are an expert AI researcher reviewing a hand pose estimation project. Please read the complete project documentation provided below and provide comprehensive, research-backed feedback.

### Project Context

**Project Asha** is a real-time EMG-to-hand-pose system for prosthetic control and gesture recognition. The ultimate goal is to predict 3D hand poses from surface electromyography (EMG) muscle signals, enabling camera-free hand tracking.

**Current Phase**: Training a transformer-based neural network (Image → MANO parameters) on the FreiHAND dataset. The project has completed:
- MediaPipe baseline (v0)
- MANO IK optimization pipeline (v1) - achieved 9.71mm MPJPE @ 25fps
- EMG integration + data recording (v2)
- ResNet training experiments (v1, v2) - plateaued at ~47mm MPJPE
- Transformer training (v1) - currently training, early epochs show ~11.4mm joint loss but total loss plateauing

### Your Task

Please review the complete `CLAUDE.md` documentation provided below and provide feedback on:

#### 1. Architecture & Training Strategy
- Is the transformer architecture (HandFormer-style) appropriate for this task?
- Are the hyperparameters (learning rate, loss weights, gradient clipping) well-chosen?
- Is the loss function (MANOLoss with parameter + joint + regularization) optimal?
- Are there any architectural improvements or alternatives you'd recommend?

#### 2. Training Issues & Solutions
- The transformer_v1 training shows stable but plateauing loss (~2.42 total, ~0.114 joint = 11.4mm MPJPE) after 5 epochs. Is this expected?
- The previous ResNet models plateaued at ~47mm MPJPE. What could cause this?
- Are there training techniques (data augmentation, curriculum learning, etc.) that could help?

#### 3. Loss Function Design
- The project uses MANOLoss: `lambda_param * param_loss + lambda_joint * joint_loss + lambda_reg * reg_loss`
- Current weights: `lambda_param=0.5`, `lambda_joint=20.0`, `lambda_reg=0.001`
- Is this weighting scheme appropriate? Should joint loss dominate this much?
- How does this compare to other hand pose estimation losses (e.g., HandFormer, MediaPipe, etc.)?

#### 4. Pose Diversity vs. MPJPE Trade-off
- The project goal emphasizes pose diversity and classifiability for EMG regression, not just minimizing MPJPE
- Is MPJPE the right metric for this use case?
- Should the loss function or evaluation metrics be adjusted to prioritize diversity?

#### 5. Dataset Strategy
- Currently using FreiHAND (130K images, 32K poses)
- Considering DexYCB (582K frames) and HO-3D (103K frames) for future training
- Is FreiHAND sufficient for initial training? When should they diversify?
- Are there other datasets they should consider?

#### 6. Research Alignment
- The project references: HandFormer (12.33mm on FreiHAND), emg2pose (EMG benchmark), NeuroPose (EMG wearables)
- Are there other recent papers (2023-2024) they should be aware of?
- Is their approach aligned with state-of-the-art methods?

#### 7. Technical Recommendations
- Any code quality or implementation concerns?
- Are there common pitfalls in transformer training for hand pose they should avoid?
- Any optimization opportunities (inference speed, model size, etc.)?

#### 8. Next Steps
- What should they prioritize next?
- Should they continue training transformer_v1 or try different hyperparameters?
- What experiments would be most valuable?

### Review Format

Please structure your feedback as:

1. **Executive Summary** (2-3 sentences)
2. **Architecture Assessment** (with citations if possible)
3. **Training Strategy Feedback** (specific recommendations)
4. **Loss Function Analysis** (comparison to literature)
5. **Dataset & Evaluation Recommendations**
6. **Research Gaps** (papers/methods they should know about)
7. **Actionable Next Steps** (prioritized list)

### Important Notes

- Be specific and actionable - avoid generic advice
- Cite papers or methods when making recommendations
- Consider the EMG end-goal (not just image→pose accuracy)
- Acknowledge what's working well, not just problems
- If you identify issues, suggest concrete solutions

---

## Instructions for Use

1. Copy the entire contents of `CLAUDE.md` from the project repository
2. Paste it below this prompt
3. Send to your chosen LLM (Grok, ChatGPT, Gemini, etc.)
4. The LLM will review the documentation and provide comprehensive feedback

---

## Example Usage

```
[Paste the prompt above]

---

[Paste CLAUDE.md contents here]
```

---

**Note**: This prompt is designed to work with any LLM. For best results:
- Use GPT-4, Claude Sonnet/Opus, or Gemini Ultra for most comprehensive feedback
- Grok may provide good research paper recommendations
- ChatGPT may provide good code/implementation feedback
- Consider asking multiple LLMs and comparing their responses

