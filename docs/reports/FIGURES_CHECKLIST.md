# CIS6800 Final Report - Figures Checklist

## Required Figures (All Present in docs/figures/)

### IK Validation Experiments (Midterm Work)
- [x] `exp1_loss_ablation.png` - Loss ablation bar chart (Slide 6, Fig. 1)
- [x] `exp4_per_joint.png` - Per-joint error analysis (Slide 8, Fig. 2)

### EMG Training Results (New Work)
- [x] `comparison_v4_vs_v5.png` - v4 vs v5 bar chart (Slide 10, Fig. 3)
- [x] `v5_training.png` - v5 training curve (Slide 11, Fig. 4)
- [x] `version_timeline.png` - Project evolution timeline (Section III, Fig. 5)
- [x] `comparison_state_of_art.png` - SOTA comparison (Section VI, Fig. 6)

## Figure Usage in LaTeX

```latex
% Section III - Methodology
\includegraphics[width=\columnwidth]{../figures/version_timeline.png}

% Section V-A - Experiment 1
\includegraphics[width=0.9\columnwidth]{../figures/exp1_loss_ablation.png}

% Section V-D - Experiment 4
\includegraphics[width=0.9\columnwidth]{../figures/exp4_per_joint.png}

% Section V-E - Experiment 5
\includegraphics[width=\columnwidth]{../figures/comparison_v4_vs_v5.png}

% Section V-F - Experiment 6
\includegraphics[width=\columnwidth]{../figures/v5_training.png}

% Section VI - Results Summary
\includegraphics[width=\columnwidth]{../figures/comparison_state_of_art.png}
```

## Compilation Instructions for Overleaf

1. **Upload files to Overleaf:**
   - Upload `CIS6800_Final.tex` to project root
   - Create a `figures/` folder in Overleaf
   - Upload all 6 required PNG files to the `figures/` folder

2. **Verify paths:**
   - LaTeX file uses `../figures/filename.png`
   - This works if .tex is in `reports/` and figures are in `figures/`
   - OR adjust paths to `figures/filename.png` if both are in root

3. **Compile:**
   - Compiler: pdfLaTeX
   - Main document: CIS6800_Final.tex
   - Should compile without errors!

## Quick Fix if Figures Don't Show

If figures don't appear, try these path formats in Overleaf:

**Option 1** (if .tex and figures/ are in root):
```latex
\includegraphics[width=\columnwidth]{figures/comparison_v4_vs_v5.png}
```

**Option 2** (if .tex is in reports/, figures/ in root):
```latex
\includegraphics[width=\columnwidth]{../figures/comparison_v4_vs_v5.png}
```

**Option 3** (absolute, always works):
```latex
\graphicspath{{../figures/}}
% Then use:
\includegraphics[width=\columnwidth]{comparison_v4_vs_v5.png}
```

## Summary

✅ All 6 required figures exist in `docs/figures/`
✅ All figure paths updated in LaTeX file
✅ All experiments (1-7) included with proper references
✅ Bibliography complete with 14 references
✅ Ready to compile in Overleaf!

**Total pages estimate:** 8-9 pages (IEEE conference format)
**Total figures:** 6 (all created and ready)
**Total tables:** 7 (all inline in LaTeX)
