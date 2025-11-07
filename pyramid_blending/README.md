# Image Pyramid Blending

손바닥과 눈 이미지를 Image Pyramid를 이용하여 자연스럽게 합성하는 프로젝트입니다.

## 프로젝트 개요

- **목표**: Gaussian Pyramid와 Laplacian Pyramid를 이용한 손바닥-눈 이미지 합성
- **기술**:
  - Gaussian Pyramid (OpenCV 및 Raw Convolution)
  - Laplacian Pyramid
  - Multi-level Blending
  - LAB Color Space Blending
- **결과물**: 완전 자동화된 파이프라인 + 시각화 + 성능 분석

## 폴더 구조

```
pyramid_blending/
├── input/                      # 입력 이미지
│   ├── hand_raw.jpg            # 손바닥 이미지
│   └── eye_raw.jpg             # 눈 이미지
├── output/                     # 출력 결과 (자동 생성)
│   ├── preprocessed/           # 전처리된 이미지
│   ├── pyramids/               # Pyramid 레벨 이미지들
│   │   ├── hand_gaussian/      # 손 Gaussian pyramid
│   │   ├── eye_gaussian/       # 눈 Gaussian pyramid
│   │   ├── mask_gaussian/      # 마스크 Gaussian pyramid
│   │   ├── hand_laplacian/     # 손 Laplacian pyramid
│   │   ├── eye_laplacian/      # 눈 Laplacian pyramid
│   │   └── blend_laplacian/    # 블렌딩된 Laplacian pyramid
│   ├── blending_results/       # 최종 합성 결과
│   ├── visualization/          # 비교 시각화
│   ├── validation/             # 검증 리포트
│   └── reports/                # 메트릭 및 로그
├── src/                        # 소스 코드
│   ├── main.py                 # 메인 실행 파일
│   ├── preprocessing.py
│   ├── pyramid_generation.py
│   ├── blending.py
│   ├── reconstruction.py
│   ├── comparison.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── validation.py
│   └── utils.py
├── requirements.txt            # 의존성 패키지
├── run.py                      # 실행 스크립트
└── README.md                   # 이 파일
```

## 설치 방법

1. **Python 환경 준비** (Python 3.7 이상 권장)

2. **의존성 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. **입력 이미지 준비**
   - 손바닥 이미지를 `input/hand_raw.jpg`에 저장
   - 눈 이미지를 `input/eye_raw.jpg`에 저장

2. **프로그램 실행**
   ```bash
   python run.py
   ```

3. **결과 확인**
   - `output/` 폴더에 모든 결과가 저장됩니다
   - `output/blending_results/` - 최종 합성 이미지
   - `output/visualization/` - 비교 그래프
   - `output/reports/` - 성능 메트릭 및 분석 리포트

## 주요 기능

### 1. 이미지 전처리
- 손바닥 이미지: 640×480으로 리사이즈
- 눈 이미지: 120×90으로 크롭 및 리사이즈
- 타원형 마스크 생성 (Gaussian blur 적용)

### 2. Pyramid 생성
- **Gaussian Pyramid**: OpenCV 및 Raw Convolution 두 가지 방식
- **Laplacian Pyramid**: 각 레벨별 디테일 정보 추출
- 6단계 pyramid (레벨 0-5: 640×480 → 20×15)

### 3. Blending 방법
- **Direct Blending**: 단순 알파 블렌딩
- **Pyramid Blending**: 레벨별 재구성 (0-5 레벨)
  - 0level: 완전 재구성 (가장 선명)
  - 5level: 기본 레벨만 (가장 블러)
- **LAB Blending**: LAB 색공간에서 블렌딩

### 4. 성능 평가
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)

### 5. 시각화
- Pyramid 레벨별 비교
- Blending 방법별 비교
- 품질 메트릭 그래프
- 명도 히스토그램
- Gaussian/Laplacian Pyramid 상세 레이아웃

### 6. 검증 파이프라인
- Gaussian Pyramid 검증 (크기, 값 범위, 분산 감소)
- Laplacian Pyramid 검증 (재구성 정확성, 특성)
- OpenCV vs Raw 구현 비교
- Blending 프로세스 검증
- 최종 결과 품질 검증

## 출력 결과물

프로그램 실행 후 다음 결과물이 생성됩니다:

### Blending Results
- `direct_blend.jpg` - Direct blending 결과
- `pyramid_blend_0level.jpg` - 레벨 0까지 재구성 (가장 선명)
- `pyramid_blend_1level.jpg` - 레벨 1까지 재구성
- `pyramid_blend_2level.jpg` - 레벨 2까지 재구성
- `pyramid_blend_3level.jpg` - 레벨 3까지 재구성
- `pyramid_blend_4level.jpg` - 레벨 4까지 재구성
- `pyramid_blend_5level.jpg` - 레벨 5만 (가장 블러, 픽셀화)
- `lab_blend_5level.jpg` - LAB 색공간 blending

### Pyramid Levels (pyramids/blend_laplacian/)
- `laplacian_level_0.jpg` ~ `laplacian_level_5.jpg` - 블렌딩된 Laplacian pyramid 각 레벨

### Visualization
- `pyramid_comparison.png` - 모든 pyramid 레벨 시각화
- `pyramid_detailed_layout.png` - Gaussian/Laplacian Pyramid 상세 레이아웃
- `blending_comparison.png` - Blending 방법 비교
- `level_comparison.png` - Pyramid 단계별 비교
- `quality_metrics.png` - 품질 메트릭 그래프
- `histogram_comparison.png` - 명도 히스토그램

### Reports
- `metrics.json` - 모든 성능 지표 (JSON 형식)
- `processing_log.txt` - 상세 처리 로그
- `analysis_summary.txt` - 분석 요약

### Validation
- `validation_report.json` - 검증 결과 (JSON 형식)
- `validation_summary.txt` - 검증 요약

## 예상 실행 결과

```
================================================================================
Image Pyramid Blending Pipeline
================================================================================

[Phase 1] Image Loading and Preprocessing
  ✓ Hand image loaded: 640×480
  ✓ Eye image cropped: 120×90 and placed on canvas
  ✓ Mask created: Ellipse + Gaussian blur (kernel=31)

[Phase 2] Gaussian Pyramid Generation (OpenCV)
  ✓ Level 0: (480, 640, 3) - Time: 0.00ms
  ✓ Level 1: (240, 320, 3) - Time: 0.88ms
  ✓ Level 2: (120, 160, 3) - Time: 0.09ms
  ✓ Level 3: (60, 80, 3) - Time: 0.03ms
  ✓ Level 4: (30, 40, 3) - Time: 0.03ms
  ✓ Level 5: (15, 20, 3) - Time: 0.02ms
  Total Memory: 4.69 MB

[Phase 3] Gaussian Pyramid Generation (Raw Convolution)
  Using Gaussian kernel: [[1,4,6,4,1], ...]
  ...

[Phase 4] Laplacian Pyramid Generation
  ✓ Hand Laplacian: 6 levels
  ✓ Eye Laplacian: 6 levels

[Phase 5] Blending
  [Comparison] Pyramid Reconstruction Levels (0-5)
    Building 6-level pyramid...
    Saving blended Laplacian pyramid levels...
    ✓ Saved 6 Laplacian levels
    Generating pyramid_blend_0level.jpg...
    Generating pyramid_blend_1level.jpg...
    ...

[Phase 6] Visualization & Reports
  ✓ All visualizations saved
  ✓ Detailed pyramid layout saved

[VALIDATION PHASE] Pyramid Blending Algorithm Verification
  [Phase 1] Gaussian Pyramid 검증...
  [Phase 2] Laplacian Pyramid 검증...
  [Phase 3] Blending 프로세스 검증...
  [Phase 4] Reconstruction 검증...
  [Phase 5] 최종 결과 품질 검증...
  ✓ Validation Complete!

✅ All completed successfully!
```

## 기술적 특징

- **데이터 타입**: 내부 처리는 float32 [0, 1], 저장은 uint8 [0, 255]
- **시간 측정**: 각 단계별 처리 시간 측정 및 로깅
- **메모리 효율**: NumPy 벡터화 연산 사용
- **에러 처리**: 이미지 크기 불일치 자동 보정

## 참고사항

- 권장 입력 이미지 크기: 손바닥 (최소 640×480), 눈 (최소 120×90)
- 처리 시간: 약 5-10초 (시스템 성능에 따라 다름)
- OpenCV vs Raw Convolution: Raw 방식이 더 느리지만 알고리즘 이해에 유용

## 라이센스

이 프로젝트는 교육 목적으로 작성되었습니다.
