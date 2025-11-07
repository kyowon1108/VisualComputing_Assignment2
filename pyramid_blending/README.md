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
│   ├── blending_results/       # 최종 합성 결과
│   ├── visualization/          # 비교 시각화
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
- 5단계 pyramid (640×480 → 40×30)

### 3. Blending 방법
- **Direct Blending**: 단순 알파 블렌딩
- **Pyramid Blending**: 3/5/6 단계 pyramid 블렌딩
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

## 출력 결과물

프로그램 실행 후 다음 결과물이 생성됩니다:

### Blending Results
- `direct_blend.jpg` - Direct blending 결과
- `pyramid_3level.jpg` - 3단계 pyramid blending
- `pyramid_5level.jpg` - 5단계 pyramid blending (권장)
- `pyramid_6level.jpg` - 6단계 pyramid blending
- `lab_blend_5level.jpg` - LAB 색공간 blending

### Visualization
- `pyramid_comparison.png` - 모든 pyramid 레벨 시각화
- `blending_comparison.png` - Blending 방법 비교
- `level_comparison.png` - Pyramid 단계별 비교
- `quality_metrics.png` - 품질 메트릭 그래프
- `histogram_comparison.png` - 명도 히스토그램

### Reports
- `metrics.json` - 모든 성능 지표 (JSON 형식)
- `processing_log.txt` - 상세 처리 로그
- `analysis_summary.txt` - 분석 요약

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
  ✓ Level 1: (240, 320, 3) - Time: 3.45ms
  ✓ Level 2: (120, 160, 3) - Time: 1.23ms
  ✓ Level 3: (60, 80, 3) - Time: 0.45ms
  ✓ Level 4: (30, 40, 3) - Time: 0.15ms
  Total Memory: 4.20 MB

[Phase 3] Gaussian Pyramid Generation (Raw Convolution)
  Using Gaussian kernel: [[1,4,6,4,1], ...]
  ...

[Phase 4] Laplacian Pyramid Generation
  ✓ Hand Laplacian: 5 levels
  ✓ Eye Laplacian: 5 levels

[Phase 5] Blending
  ...

[Phase 6] Visualization & Reports
  ✓ All visualizations saved

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
