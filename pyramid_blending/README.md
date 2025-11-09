# Image Pyramid Blending

Python implementation of multi-scale image blending using Gaussian and Laplacian pyramids.

## Features

- **Alpha Blending**: Direct weighted blending
- **Pyramid Blending**: Multi-scale Laplacian pyramid blending (6 levels)
- Comprehensive quality metrics (SSIM, MSE, PSNR)
- Multiple analysis tools (ROI, boundary, artifacts)

## Project Structure

```
pyramid_blending/
├── src/                    # Source code
│   ├── preprocessing.py
│   ├── pyramid_generation.py
│   ├── blending.py
│   ├── reconstruction.py
│   ├── metrics.py
│   └── ...
├── scripts/                # Analysis scripts
│   ├── run_ablation_study.py
│   ├── run_profiling.py
│   ├── create_comprehensive_comparison.py
│   └── ...
├── images/                 # Input images
├── output/                 # Generated results
└── run.py                  # Main pipeline
## Usage

### Run Main Pipeline

```bash
python3 run.py
```

### Run Analysis Scripts

```bash
# Ablation study (blur kernel variation)
python3 scripts/run_ablation_study.py

# Performance profiling
python3 scripts/run_profiling.py

# Create comprehensive comparison
python3 scripts/create_comprehensive_comparison.py

# ROI-based quality analysis
python3 scripts/run_roi_analysis.py

# Boundary quality analysis  
python3 scripts/run_boundary_analysis.py

# Artifact analysis
python3 scripts/run_artifact_analysis.py

# Heatmap analysis
python3 scripts/run_heatmap_analysis.py
```

## Output

All results are saved to `output/`:

- `blending_results/` - Blending output images
- `pyramids/` - Gaussian and Laplacian pyramid levels
- `visualization/` - Comparison visualizations
- `reports/` - Metrics and analysis reports
- `validation/` - Validation reports

## Blending Methods

### 1. Alpha Blending (Direct)
- Simple weighted average: `I = (1-α)·I₁ + α·I₂`
- Fast but visible seams

### 2. Pyramid Blending
- Multi-scale decomposition using Laplacian pyramids
- Smooth transitions at boundaries
- 6 reconstruction levels (L0 to L5)
  - **L0**: Full reconstruction (best quality, SSIM ~0.99)
  - **L5**: Base level only (coarse, SSIM ~0.62)

## Requirements

```
numpy
opencv-python
scikit-image
matplotlib
psutil
```

## Results

- **Best Quality**: Pyramid to L0 (Full) - SSIM 0.9924
- **Processing Speed**: ~2-6ms per blend
- **Memory Usage**: ~150MB for full pipeline

## License

Educational project for Visual Computing course.
