# references

## core papers

**mediapipe (google, 2020)**
- real-time hand landmark detection (21 joints, 60fps)
- baseline for joint positions

**mano (romero et al., siggraph asia 2017)**
- parametric hand model (45 pose params + 10 shape params)
- linear blend skinning for anatomical accuracy
- core method for converting positions → angles

**drosakis & argyros (petra 2023)**
- mano optimization from 2d keypoints
- exact match to our method
- shows optimization competitive with learning-based approaches

## comparison papers

**handformer (jiao et al., pattern recognition letters 2024)**
- transformer + mlp for monocular hand pose
- 10.92-12.33mm error (our accuracy baseline)

**tu et al. (ieee tpami 2022)**
- consistent 3d hand reconstruction in video
- multi-term loss: 2d keypoints + motion + texture + shape
- justifies our temporal smoothness loss

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

## ground truth & weak supervision

**cai et al. (eccv 2018)**
- weak supervision from monocular rgb
- depth regularizer
- relevant for low-cost ground truth generation

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
- 193 users, 370 hours, 16ch emg, 26-camera mocap
- shows need for high-quality pose ground truth
- validates our application
