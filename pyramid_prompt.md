# Image Pyramid Blending 과제 - Claude Code 구현 프롬프트

## 프로젝트 개요
**목표**: Image Pyramid를 이용한 손바닥 이미지와 눈 이미지 합성  
**기술**: Gaussian Pyramid, Laplacian Pyramid, Multi-level Blending  
**결과물**: 완전 자동화된 파이프라인 + 시각화 + 성능 분석

---

## 폴더 구조 및 파일 설명

```
pyramid_blending/
│
├── input/
│   ├── hand_raw.jpg          # 원본 손바닥 이미지 (사용자 촬영)
│   └── eye_raw.jpg           # 원본 눈 이미지 (사용자 촬영)
│
├── output/
│   ├── preprocessed/
│   │   ├── hand_640x480.jpg  # 전처리된 손바닥 (640×480)
│   │   ├── eye_120x90.jpg    # 전처리된 눈 (120×90)
│   │   ├── mask.png          # 생성된 마스크 (타원형+blur)
│   │   └── mask_ellipse.png  # 마스크 (blur 전)
│   │
│   ├── pyramids/
│   │   ├── hand_gaussian/
│   │   │   ├── level_0.png   # 640×480
│   │   │   ├── level_1.png   # 320×240
│   │   │   ├── level_2.png   # 160×120
│   │   │   ├── level_3.png   # 80×60
│   │   │   └── level_4.png   # 40×30
│   │   ├── eye_gaussian/     # (동일 구조)
│   │   ├── mask_gaussian/    # (동일 구조)
│   │   ├── hand_laplacian/   # (동일 구조)
│   │   └── eye_laplacian/    # (동일 구조)
│   │
│   ├── blending_results/
│   │   ├── direct_blend.jpg       # Direct (Alpha) Blending 결과
│   │   ├── pyramid_5level.jpg     # Pyramid Blending (5단계)
│   │   ├── pyramid_3level.jpg     # Pyramid Blending (3단계)
│   │   ├── pyramid_6level.jpg     # Pyramid Blending (6단계)
│   │   └── lab_blend_5level.jpg   # LAB 색공간 Blending (5단계)
│   │
│   ├── visualization/
│   │   ├── pyramid_comparison.png     # 모든 레벨 시각화 비교
│   │   ├── blending_comparison.png   # Direct vs Pyramid 비교
│   │   ├── level_comparison.png      # 3/5/6단계 비교
│   │   ├── histogram_comparison.png  # 명도 히스토그램 비교
│   │   └── quality_metrics.png       # SSIM/MSE 그래프
│   │
│   └── reports/
│       ├── metrics.json              # 모든 성능 지표 (JSON)
│       ├── processing_log.txt        # 프로세싱 로그
│       └── analysis_summary.txt      # 분석 요약
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # 메인 실행 파일
│   ├── preprocessing.py              # 이미지 전처리
│   ├── pyramid_generation.py         # Pyramid 생성 (OpenCV + Raw)
│   ├── blending.py                   # Blending 알고리즘
│   ├── reconstruction.py             # Pyramid Reconstruction
│   ├── comparison.py                 # 비교 실험 (3/5/6 level, RGB/LAB)
│   ├── visualization.py              # 시각화 함수
│   ├── metrics.py                    # 품질 메트릭 (SSIM, MSE)
│   └── utils.py                      # 유틸리티 함수
│
└── requirements.txt                   # 의존성 패키지
```

---

## 각 모듈 상세 설명

### **1. `main.py` - 메인 파이프라인**

**역할**: 전체 프로세싱 파이프라인을 조율

**구현 요구사항**:
```python
def main():
    """
    전체 파이프라인 실행
    
    Steps:
    1. 이미지 로드 및 전처리
    2. Mask 생성
    3. Gaussian Pyramid 생성 (hand, eye, mask) - OpenCV
    4. Gaussian Pyramid 생성 (hand, eye) - Raw convolution
    5. Laplacian Pyramid 생성
    6. Direct Blending
    7. Pyramid Blending (3/5/6 level)
    8. LAB 색공간 Blending
    9. 성능 메트릭 계산
    10. 시각화 및 결과 저장
    """
```

**프로세싱 출력 예시**:
```
[Phase 1] 이미지 로드 및 전처리
  ✓ Hand image loaded: 640×480
  ✓ Eye image cropped: 120×90
  ✓ Mask created: Ellipse + Gaussian blur (kernel=31)

[Phase 2] Gaussian Pyramid Generation (OpenCV)
  ✓ Level 0: (480, 640, 3) - Time: 12.34ms
  ✓ Level 1: (240, 320, 3) - Time: 3.45ms
  ✓ Level 2: (120, 160, 3) - Time: 1.23ms
  ✓ Level 3: (60, 80, 3) - Time: 0.45ms
  ✓ Level 4: (30, 40, 3) - Time: 0.15ms
  Total Memory: 4.2 MB

[Phase 3] Gaussian Pyramid Generation (Raw Convolution)
  Using Gaussian kernel: [[1,4,6,4,1], ...]
  ✓ Level 0: (480, 640, 3) - Time: 234.56ms
  ✓ Level 1: (240, 320, 3) - Time: 58.90ms
  ✓ Level 2: (120, 160, 3) - Time: 14.67ms
  ✓ Level 3: (60, 80, 3) - Time: 3.67ms
  ✓ Level 4: (30, 40, 3) - Time: 0.92ms
  Total Memory: 4.2 MB

[Phase 4] Laplacian Pyramid Generation
  ✓ Hand Laplacian: 5 levels
  ✓ Eye Laplacian: 5 levels

[Phase 5] Blending
  Direct Blending: SSIM = 0.6234, MSE = 1523.45
  Pyramid (3-level): SSIM = 0.7543, MSE = 1123.22
  Pyramid (5-level): SSIM = 0.8234, MSE = 892.11
  Pyramid (6-level): SSIM = 0.8156, MSE = 945.67
  LAB Blend (5-level): SSIM = 0.8145, MSE = 978.34

[Phase 6] Visualization & Report
  ✓ Pyramid comparison saved
  ✓ Blending comparison saved
  ✓ Metrics report saved
  ✓ Processing log saved

✅ All completed successfully!
```

---

### **2. `preprocessing.py` - 이미지 전처리**

**역할**: 원본 이미지 → 표준 크기 + Mask 생성

```python
def load_and_preprocess(hand_path, eye_path):
    """
    Input: 손/눈 원본 이미지
    Output:
      - hand_img: (480, 640, 3) dtype=float32
      - eye_img: (480, 640, 3) dtype=float32 (눈만 데이터, 나머지 검정)
    
    Steps:
    1. 손 이미지: center crop → 640×480 resize (비율 유지)
    2. 눈 이미지: crop (머리카락 제외) → 120×90 resize (비율 유지)
    3. 눈을 (480, 640) 캔버스의 (480, 240) 위치에 배치
    """

def create_mask(shape=(480, 640)):
    """
    Mask 생성
    - Shape: (480, 640)
    - Ellipse 중심: (480, 240)
    - Ellipse 축: (60, 45)
    - Gaussian Blur: kernel=31, sigma=auto
    - Output range: [0, 1]
    
    과정:
    1. 흰색 배경 (모두 0)
    2. 타원형 영역만 1로 설정
    3. Gaussian blur 적용 (feathering)
    """
```

---

### **3. `pyramid_generation.py` - Pyramid 생성**

**역할**: Gaussian Pyramid 생성 (2가지 방식)

```python
def gaussian_pyramid_opencv(image, levels=5):
    """
    Method A: OpenCV 사용 (빠름, 참조용)
    
    Args:
        image: (H, W, 3)
        levels: pyramid 단계 수
    
    Output:
        gp: list of (H, W, 3) arrays
        - gp[0]: (480, 640, 3)
        - gp[1]: (240, 320, 3)
        - ...
        - gp[4]: (30, 40, 3)
    
    Implementation:
        gp[0] = image
        for i in range(1, levels):
            gp[i] = cv2.pyrDown(gp[i-1])
    
    Note: 각 레벨 이미지 저장 (output/pyramids/)
    """

def gaussian_pyramid_raw(image, levels=5):
    """
    Method B: Raw Convolution (상세 분석)
    
    Args:
        image: (H, W, 3)
        levels: pyramid 단계 수
    
    Gaussian Kernel (5×5):
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]] / 256
    
    Process per level:
        1. Gaussian convolution (same padding)
        2. Stride=2 subsampling
        3. Output 크기 정확성 검증
    
    Output:
        gp: list of (H, W, 3) arrays
    
    Note: 각 단계별 처리 시간 측정 및 로깅
    """
```

---

### **4. `reconstruction.py` - Reconstruction**

**역할**: Laplacian Pyramid → 원본 크기 복원

```python
def reconstruct_from_laplacian(lap_pyramid):
    """
    Laplacian Pyramid → 원본 크기 이미지 복원
    
    Args:
        lap_pyramid: [L0 (480×640), L1 (240×320), ..., L4 (30×40)]
    
    Algorithm:
        result = L[4]  # 가장 작은 레벨부터 시작
        for level in range(3, -1, -1):
            result = cv2.pyrUp(result)  # 2배 크기로 upsample
            result = result + L[level]  # Laplacian 덧셈
    
    Output:
        reconstructed: (480, 640, 3)
    
    Validation:
        - 각 단계에서 크기 정확성 확인
        - 원본과의 차이 (error) 측정
    """
```

---

### **5. `blending.py` - Blending 알고리즘**

**역할**: 다양한 blending 방식 구현

```python
def direct_blending(hand, eye, mask):
    """
    Method 1: Direct (Alpha) Blending
    
    Formula:
        output = hand × (1 - mask) + eye × mask
    
    Args:
        hand: (480, 640, 3)
        eye: (480, 640, 3)
        mask: (480, 640, 1)
    
    Output:
        blended: (480, 640, 3)
    """

def pyramid_blending(hand_lap, eye_lap, mask_gp, levels=5):
    """
    Method 2: Pyramid Blending (Multi-level)
    
    Args:
        hand_lap: list of 5 Laplacian levels
        eye_lap: list of 5 Laplacian levels
        mask_gp: list of 5 Gaussian mask levels
        levels: pyramid 단계 수 (3/5/6)
    
    Process:
        for i in range(levels):
            blended_lap[i] = hand_lap[i] × (1 - mask_gp[i]) 
                           + eye_lap[i] × mask_gp[i]
    
    Output:
        blended_lap: list of 5 blended Laplacian levels
    
    Note: 다시 reconstruction 필요
    """

def lab_blending(hand_lap, eye_lap, mask_gp, levels=5):
    """
    Method 3: LAB 색공간 Blending (선택)
    
    Process:
        1. Hand LAB pyramid 생성 (L/a/b 채널 분리)
        2. Eye LAB pyramid 생성
        3. L 채널만 blending: L_blended = Hand_L × (1-mask) + Eye_L × mask
        4. a/b 채널은 Hand 유지
        5. LAB → RGB 변환
    
    Output:
        lab_blended: (480, 640, 3) RGB 이미지
    """
```

---

### **6. `metrics.py` - 품질 메트릭 계산**

**역할**: SSIM, MSE 계산 및 로깅

```python
def calculate_metrics(result, reference):
    """
    Args:
        result: 합성된 이미지 (480, 640, 3)
        reference: 참조 이미지 (480, 640, 3)
    
    Returns:
        metrics: dict
            - ssim: Structural Similarity Index (0~1, 높을수록 좋음)
            - mse: Mean Squared Error (낮을수록 좋음)
            - psnr: Peak Signal-to-Noise Ratio (높을수록 좋음)
    
    Implementation:
        - skimage.metrics.structural_similarity (SSIM)
        - sklearn.metrics.mean_squared_error (MSE)
        - PSNR = 20 × log10(MAX_PIXEL_VALUE / sqrt(MSE))
    """

def save_metrics_report(metrics_dict, output_path):
    """
    모든 실험 결과를 JSON으로 저장
    
    Format:
    {
        "direct_blending": {"ssim": 0.62, "mse": 1523.45, ...},
        "pyramid_3level": {"ssim": 0.75, "mse": 1123.22, ...},
        "pyramid_5level": {"ssim": 0.82, "mse": 892.11, ...},
        "pyramid_6level": {"ssim": 0.82, "mse": 945.67, ...},
        "lab_blend_5level": {"ssim": 0.81, "mse": 978.34, ...}
    }
    """
```

---

### **7. `visualization.py` - 시각화**

**역할**: 결과 이미지 및 그래프 생성

```python
def visualize_pyramid_levels(pyramid_dict):
    """
    모든 pyramid 레벨 시각화
    
    Output: 
        visualization/pyramid_comparison.png
        - Subplot grid: hand_gp, eye_gp, mask_gp (각 5×5)
        - 각 이미지 위에 레벨 번호, 크기 표시
    """

def visualize_blending_comparison(direct, pyramid3, pyramid5, pyramid6, lab):
    """
    Blending 방식 비교 시각화
    
    Output:
        visualization/blending_comparison.png
        - 5개 이미지를 1×5 grid로 표시
        - 각 이미지 아래에 방식명, SSIM, MSE 표시
    """

def plot_quality_metrics(metrics_dict):
    """
    SSIM/MSE 그래프
    
    Output:
        visualization/quality_metrics.png
        - 막대그래프: 각 방식별 SSIM 및 MSE 비교
    """

def plot_histogram_comparison(results):
    """
    명도 히스토그램 비교
    
    Output:
        visualization/histogram_comparison.png
        - Hand original, Direct blend, Pyramid blend의 명도 히스토그램
    """
```

---

### **8. `comparison.py` - 비교 실험**

**역할**: 다양한 조건에서 실험 실행

```python
def compare_pyramid_levels():
    """
    Pyramid 레벨 비교: 3 vs 5 vs 6
    
    Process:
        1. 3단계 pyramid 생성 및 blending
        2. 5단계 pyramid 생성 및 blending
        3. 6단계 pyramid 생성 및 blending
        4. 각 결과 저장 및 메트릭 계산
    
    Output:
        - output/blending_results/pyramid_3level.jpg
        - output/blending_results/pyramid_5level.jpg
        - output/blending_results/pyramid_6level.jpg
        - metrics_dict 반환
    """

def compare_direct_vs_pyramid():
    """
    Direct blending vs Pyramid blending (5단계)
    
    Output:
        - 메트릭 비교
        - visualization/blending_comparison.png
    """

def compare_color_spaces():
    """
    RGB vs LAB 색공간 비교
    
    Process:
        1. RGB blending (기본)
        2. LAB blending (L 채널만 blend)
        3. 결과 및 메트릭 비교
    
    Output:
        - metrics 비교
        - visualization에 결과 포함
    """
```

---

### **9. `utils.py` - 유틸리티**

**역할**: 공용 함수

```python
def create_output_directories():
    """폴더 구조 자동 생성"""

def save_image(image, path):
    """이미지 저장 (정규화 포함)"""

def load_image(path):
    """이미지 로드 및 정규화 ([0,1] 또는 [0,255])"""

def normalize_image(image):
    """이미지 정규화 (dtype 변환)"""

def measure_time(func):
    """실행 시간 측정 데코레이터"""

def log_message(message, level="INFO"):
    """로깅 함수"""
```

---

## 구현 세부 요구사항

### **1. 데이터 타입 및 정규화**
- **입력**: uint8 ([0, 255])
- **내부 처리**: float32 ([0, 1] 또는 [0, 255] 일관성)
- **출력**: uint8 ([0, 255])
- **중간 이미지 저장**: PNG (uint8)

### **2. 시간 측정**
- 각 주요 단계의 실행 시간 측정 및 로깅
- OpenCV vs Raw 비교
- Pyramid 레벨별 처리 시간 추적

### **3. 에러 처리**
- 이미지 크기 불일치 시 자동 보정
- 메모리 부족 예방
- 파일 입출력 오류 처리

### **4. 로깅 및 리포트**
- 각 단계별 프로세싱 로그 저장 (txt)
- 최종 메트릭 리포트 (JSON)
- 분석 요약 생성

---

## 실행 예시

```bash
# 기본 실행
python src/main.py

# 결과 확인
ls -R output/
# → preprocessed/, pyramids/, blending_results/, visualization/, reports/

# 최종 결과
# output/blending_results/pyramid_5level.jpg (최종 합성 이미지)
# output/reports/metrics.json (성능 지표)
# output/visualization/*.png (모든 비교 이미지)
```

---

## 주요 검증 포인트

✅ **Gaussian Pyramid**
- 각 레벨 크기: 정확히 1/2씩 감소
- 마지막 레벨: (30, 40, 3) 또는 (30, 40, 1)
- 정보 손실: 최소화

✅ **Laplacian Pyramid**
- 각 L[i] = G[i] - upsample(G[i+1]) 정확성
- Reconstruction 후 원본과의 차이 < 0.1% 픽셀값

✅ **Blending**
- 경계선 (mask ellipse): 부드러운 transition
- 눈 영역: 명확하게 보임
- 손바닥: 자연스러운 색상 유지

✅ **메트릭**
- SSIM: Direct < Pyramid (향상도 명확)
- 3 < 5 ≈ 6 (레벨별 차이 확인)
- 색공간: RGB ≈ LAB (큰 차이 없음 또는 명확한 장점)

---

## 추가 참고사항

### **마스크 타원형 좌표**
- 중심: (480, 240) → OpenCV는 (y, x) 기반이므로 (240, 480) 사용
- 축: (60, 45) → (축_x, 축_y)

### **이미지 저장 경로**
- 모든 중간 결과 저장 (리포트용)
- 최종 결과만 보기 쉽게 구분

### **성능 최적화**
- NumPy 벡터화 연산 사용
- 불필요한 복사 최소화

---

## 예상 산출물

✅ `output/preprocessed/` - 전처리된 입력 이미지  
✅ `output/pyramids/` - 모든 pyramid 레벨 이미지  
✅ `output/blending_results/` - 최종 합성 결과 (5가지)  
✅ `output/visualization/` - 비교 그래프 및 시각화  
✅ `output/reports/` - 메트릭 및 로그  

**모두 이 결과물들이 PDF 리포트에 포함될 예정입니다.**

---

## 완료 체크리스트

- [ ] 모든 파이썬 파일 구현 완료
- [ ] 각 단계별 프로세싱 로그 출력 확인
- [ ] 모든 중간 이미지 저장 확인
- [ ] 성능 메트릭 계산 및 저장 확인
- [ ] 시각화 이미지 생성 확인
- [ ] 최종 리포트 JSON 생성 확인
