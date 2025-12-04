# references

## core papers

**mediapipe (google, 2020)**
- real-time hand landmark detection (21 joints, 60fps)
- baseline for joint positions
- our use: initial hand detection before mano ik

**mano (romero et al., siggraph asia 2017)**
- parametric hand model (45 pose params + 10 shape params)
- linear blend skinning for anatomical accuracy
- our use: core method for converting positions → angles via ik optimization

**drosakis & argyros (petra 2023)**
- mano optimization from 2d keypoints
- exact match to our method (extend with bone direction + temporal losses)
- shows optimization competitive with learning-based approaches
- our use: methodology baseline, validates optimization approach

## comparison papers

**handformer (jiao et al., pattern recognition letters 2024)**
- transformer + mlp for monocular hand pose
- 10.92mm (stereo), 12.33mm (freihand)
- our use: accuracy baseline (our 9.71mm competitive)

**tu et al. (ieee tpami 2022)**
- consistent 3d hand reconstruction in video
- multi-term loss: 2d keypoints + motion + texture + shape
- our use: justifies temporal smoothness loss

**guo et al. (ieee tcsvt 2022)**
- cnn + gcn + attention
- skeleton-aware feature interaction

**jiang et al. (cvpr 2023)**
- a2j-transformer for 3d hand pose
- class relevant (transformers)

**gao et al. (neurocomputing 2021)**
- transformer-based ik (tiknet)
- mano parameter regression
- future work consideration

**kalshetti & chaudhuri (ieee tpami 2025)**
- differentiable rendering + icp optimization
- optimization approach relevant

## datasets

**freihand (zimmermann et al., eccv 2019)**
- 32,560 unique samples × 4 views = 130k total
- multi-view triangulation ground truth (~5mm accuracy)
- 224×224 rgb images
- our use: public dataset evaluation (experiment 5 validation)
- results: 20% detection rate (challenging poses expected), 16.21mm mean error on detected

**ho-3d (hampali et al., cvpr 2020 - honnotate paper)**
- 103,462 annotated frames
- rgb-d hand-object interaction
- 640×480 resolution, severe occlusion challenge
- annotations: 3d joint positions (handJoints3D), may require mano fitting
- our use: public dataset evaluation with occlusion (experiment 5)
- results: 5% detection rate (occlusion heavy), 17.64mm mean error on detected
- download: https://github.com/shreyashampali/ho3d or kaggle datasets download -d marcmarais/ho3d-v3

**dexycb (chao et al., cvpr 2021)**
- 582k rgb-d frames, 1,000 sequences, 10 subjects, 8 views
- hand-object grasping interactions with mano parameters
- mano annotations: direct mano parameters (theta, beta) with fixed shape per subject
- best for: pose diversity, grasping poses, classifiable gestures
- download: https://dex-ycb.github.io/
  - option 1: single file `dex-ycb-20210415.tar.gz` (119gb) - recommended
  - option 2: multiple files (10 subject files ~12gb each + calibration + models)
- toolkit: https://github.com/NVlabs/dex-ycb-toolkit
- license: cc by-nc 4.0 (non-commercial use)
- paper: docs/references/dexycb2021.pdf
- our use: best addition for pose diversity (582k frames vs 103k ho-3d)

## ground truth & weak supervision

**cai et al. (eccv 2018)**
- weak supervision from monocular rgb
- depth regularizer
- our use: relevant for low-cost ground truth generation approach

**cai et al. (ieee tpami 2020)**
- cvae + weak supervision with synthetic data
- bridges sim-to-real gap

## advanced cv techniques (class topics)

**spurr et al. (iccv 2021)**
- self-supervised 3d hand pose via contrastive learning (peclr)
- 14.5% pa-epe improvement on freihand
- class relevant (self-supervised learning)

**handdiff (cheng et al., cvpr 2024)**
- diffusion models for hand pose
- rgb-d, iterative denoising
- class relevant (diffusion models)

## application domain

**emg2pose (meta fair, arxiv 2024)**
- 193 users, 370 hours, 16ch emg @ 2khz, 26-camera mocap
- joint angles as primary representation
- shows need for high-quality pose ground truth
- our use: validates application domain, informs v3-v4 training pipeline design
- comparison: same pipeline (mocap → angles), we use webcam + mediapipe instead

**neuropose (jiang et al., penn state 2021)**
- rnn/u-net architectures with anatomical constraints
- wearable emg for fine-grained finger motion
- our use: architecture reference for v3 training pipeline
