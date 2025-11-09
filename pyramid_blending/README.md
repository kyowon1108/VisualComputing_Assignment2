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
├── src/                    # 핵심 소스 코드
│   ├── preprocessing.py        # 이미지 전처리 및 마스크 생성
│   ├── pyramid_generation.py   # Gaussian/Laplacian 피라미드 생성
│   ├── blending.py             # 블렌딩 알고리즘
│   ├── reconstruction.py       # 피라미드 재구성
│   ├── metrics.py              # 품질 평가 지표 (SSIM, MSE, PSNR)
│   ├── roi_analysis.py         # ROI 기반 정량적 분석
│   ├── boundary_analysis.py    # 경계 품질 분석
│   ├── visualization.py        # 시각화 도구
│   └── main.py                 # 메인 실행 로직
│
├── scripts/                # 분석 및 생성 스크립트
│   ├── run_roi_analysis.py                    # ROI 분석 실행
│   ├── run_boundary_analysis.py               # 경계 분석 실행
│   ├── run_ablation_study.py                  # Ablation study
│   ├── run_profiling.py                       # 성능 프로파일링
│   ├── generate_roi_comparison_all_levels.py  # ROI 비교 이미지 생성
│   └── generate_troubleshooting_images.py     # 트러블슈팅 이미지 생성
│
├── docs/                   # 문서 (한국어/영어)
│   ├── 평가보고서_이미지포함.md               # 최종 평가 보고서 (한국어, 이미지 포함)
│   ├── 평가보고서.md                          # 평가 보고서 (한국어)
│   ├── 코드구현_품질평가_ROI분석.md           # 코드 구현 상세 설명
│   ├── TROUBLESHOOTING_KR.md                  # 트러블슈팅 가이드 (한국어)
│   ├── TROUBLESHOOTING.md                     # 트러블슈팅 가이드 (영어)
│   ├── EVALUATION_REPORT.md                   # 평가 보고서 (영어)
│   ├── pyramid_blending_explanation.md        # 피라미드 블렌딩 원리 설명
│   └── TERMINOLOGY_UPDATE_SUMMARY.md          # 용어 업데이트 요약
│
├── input/                  # 입력 이미지
│   ├── hand_raw.jpg            # 배경 이미지 (손)
│   └── eye_raw.jpg             # 삽입 객체 (눈)
│
├── output/                 # 생성된 결과물
│   ├── blending_results/       # 블렌딩 결과 이미지
│   ├── pyramids/               # Gaussian/Laplacian 피라미드
│   ├── roi_analysis/           # ROI 분석 결과
│   ├── troubleshooting/        # 트러블슈팅 시각화
│   ├── boundary_analysis/      # 경계 분석 결과
│   ├── visualization/          # 기타 시각화
│   └── reports/                # JSON 메트릭 리포트
│
├── run.py                  # 메인 실행 스크립트
├── requirements.txt        # Python 의존성
└── README.md               # 프로젝트 설명 (이 파일)
```

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
