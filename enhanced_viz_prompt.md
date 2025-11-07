# Enhanced Pyramid Visualization Prompt for Claude Code

## 추가 요구사항: Gaussian & Laplacian Pyramid 통합 시각화

---

## 📐 시각화 목표

**첨부된 예시 이미지의 스타일처럼** Gaussian Pyramid와 Laplacian Pyramid를 한 이미지에 통합 표시:

```
┌──────────────────────────────────────────────────────────────────┐
│  "gaussian pyramid" (제목)                                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Level 0: 480×640]        [Laplacian 0]  [Laplacian 0 vis]     │
│  [Full image]              [Detail map]   [Brightened version]  │
│                                                                    │
│  [Level 1: 240×320]        [Laplacian 1]  [Laplacian 1 vis]     │
│  [Downsampled]             [Detail map]   [Brightened version]  │
│                                                                    │
│  [Level 2: 120×160]        [Laplacian 2]  [Laplacian 2 vis]     │
│  [Blurred]                 [Detail map]   [Brightened version]  │
│                                                                    │
│  [Level 3: 60×80]          [Laplacian 3]  [Laplacian 3 vis]     │
│  [More blurred]            [Detail map]   [Brightened version]  │
│                                                                    │
│  [Level 4: 30×40]          [Laplacian 4]  [Laplacian 4 vis]     │
│  [Highly abstract]         [Detail map]   [Brightened version]  │
│                                                                    │
│  [Level 5: 15×20]          [Laplacian 5]  [Laplacian 5 vis]     │
│  [Base layer]              [Base layer]   [Brightened version]  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 구체적 시각화 사양

### **1. 레이아웃 구조**
- **전체 배경**: 흰색 또는 밝은 회색
- **제목**: "Gaussian Pyramid & Laplacian Pyramid Analysis"
- **컬럼 구성** (좌 → 우):
  1. **Gaussian Pyramid**: 원본 이미지 (Level 0 ~ Level 5)
  2. **Laplacian Pyramid**: Detail map (Level 0 ~ Level 5)
  3. **Laplacian Brightened**: 시각적 명도 조정 버전 (Level 0 ~ Level 5)

### **2. 각 레벨별 표시 정보**
- **레벨 번호**: "Level 0", "Level 1", ... "Level 5"
- **이미지 크기**: "(480×640)", "(240×320)", ... "(15×20)"
- **설명 텍스트**:
  - Gaussian: "Original", "1/2 downsampled", "1/4 downsampled", ...
  - Laplacian: "Detail", "Detail", ..., "Base (G5)"

### **3. 이미지 크기 조정**
- **Gaussian 레벨**: 원본 크기대로 표시 (자동 스케일링)
- **Laplacian 레벨**: Gaussian과 동일 크기로 resize하여 정렬
- **아래쪽 레벨**: 위쪽 레벨보다 작게 표시 (자연스러운 pyramid 형태)

### **4. 컬러맵 적용**
- **Gaussian**: 원본 컬러 유지 (RGB)
- **Laplacian (Raw)**: Colormap 적용
  - `cv2.COLORMAP_JET` 또는 `cv2.COLORMAP_VIRIDIS` (디테일 강조)
  - 또는 중심값 127에서의 deviation 표시 (파란색=음수, 빨간색=양수)
- **Laplacian (Brightened)**: 정규화 후 표시
  - 명도 범위: [0, 255]로 재조정하여 더 선명하게

### **5. 선과 텍스트 스타일**
- **레벨 분리선**: 흑색 점선 (horizontal lines)
- **컬럼 분리선**: 흑색 실선 (vertical lines)
- **폰트**: 검정색, 크기 10-12pt, 산세리프 (Arial)
- **레이블 위치**: 각 이미지 위쪽 또는 좌측

---

## 🎨 실제 구현 코드 패턴

### **Function: `visualize_pyramid_detailed_layout()`**

```python
def visualize_pyramid_detailed_layout(hand_gaussian_pyr, hand_laplacian_pyr, 
                                      output_path="visualization/pyramid_detailed.png"):
    """
    Gaussian Pyramid & Laplacian Pyramid 통합 시각화
    
    Args:
        hand_gaussian_pyr: list of Gaussian pyramid images [G0, G1, ..., G5]
        hand_laplacian_pyr: list of Laplacian pyramid images [L0, L1, ..., L5]
        output_path: 저장 경로
    
    Process:
        1. Figure 생성 (figsize=(24, 16)) - 큰 캔버스
        2. GridSpec 또는 SubplotSpec으로 3-column layout 구성
           - Col 0: Gaussian Pyramid
           - Col 1: Laplacian Pyramid (raw)
           - Col 2: Laplacian Pyramid (brightened/colormap)
        3. 각 레벨별 행 생성
           - Row 0: Level 0 (480×640)
           - Row 1: Level 1 (240×320)
           - ...
           - Row 5: Level 5 (15×20)
        4. 각 이미지 자동 스케일링:
           - 작은 이미지는 interpolation으로 시각적 크기 확대
           - 또는 각 행의 높이를 레벨별로 달리 설정
        5. Colormap 적용:
           - Laplacian (col 1): JET 또는 custom colormap
           - Laplacian (col 2): Normalized [0, 255]
        6. 각 이미지 아래 텍스트 추가:
           - 레벨, 크기, 설명
        7. 전체 제목 추가: "Gaussian & Laplacian Pyramid Analysis"
    
    Output:
        PNG file: visualization/pyramid_detailed.png
        - 높은 해상도 (DPI=150 이상)
        - 모든 정보 명확하게 표시
    """
```

### **핵심 구현 포인트**

1. **GridSpec 활용**:
   ```python
   fig = plt.figure(figsize=(24, 16), dpi=150)
   gs = fig.add_gridspec(nrows=6, ncols=3, hspace=0.4, wspace=0.3)
   
   for level in range(6):
       # Col 0: Gaussian
       ax_g = fig.add_subplot(gs[level, 0])
       ax_g.imshow(gaussian_pyr[level])
       ax_g.set_title(f"Level {level} (Gaussian)\n{size[level]}")
       
       # Col 1: Laplacian (Raw with colormap)
       ax_l1 = fig.add_subplot(gs[level, 1])
       ax_l1.imshow(laplacian_pyr[level], cmap='jet')
       ax_l1.set_title(f"Level {level} (Laplacian)\n{size[level]}")
       
       # Col 2: Laplacian (Brightened)
       ax_l2 = fig.add_subplot(gs[level, 2])
       lap_normalized = normalize_laplacian(laplacian_pyr[level])
       ax_l2.imshow(lap_normalized, cmap='gray')
       ax_l2.set_title(f"Level {level} (Brightened)\n{size[level]}")
   ```

2. **Laplacian 정규화 함수**:
   ```python
   def normalize_laplacian(laplacian_img, method='min_max'):
       """
       Laplacian 이미지는 음수값을 포함하므로, 시각화를 위해 정규화
       
       Method 1: Min-Max normalization (0-255)
       Method 2: Center at 127 (neg=blue, pos=red)
       Method 3: Absolute value (모든 값 양수화)
       """
   ```

3. **Colormap 선택**:
   ```python
   # Option A: Diverging colormap (권장)
   # 중심값(0)을 기준으로 음수(파란색)와 양수(빨간색)를 다르게 표시
   colormap = 'RdBu_r'  # Red-Blue reversed
   
   # Option B: Intensity colormap
   colormap = 'jet'  # 다양한 색상으로 detail 강조
   
   # Option C: Custom colormap
   # 직접 설계하여 음수/양수 시각적 구분
   ```

4. **이미지 크기 자동 조정**:
   ```python
   def get_display_size(level):
       """
       각 레벨의 표시 크기 결정
       - 기본: 원본 크기대로 표시
       - 아래 레벨이 너무 작으면, 보간으로 확대
       - 행 높이는 다르게 (위쪽 레벨이 크게)
       """
   ```

---

## 📝 추가 시각화 옵션

### **Option 1: Reconstruction 과정 시각화**
```
┌─────────────────────────────────────────┐
│  "Reconstruction Process"               │
├─────────────────────────────────────────┤
│  Level 5 (Base) → upsample + add L4     │
│         ↓                               │
│  Reconstructed L4 → upsample + add L3  │
│         ↓                               │
│  Reconstructed L3 → upsample + add L2  │
│         ↓                               │
│  Reconstructed L2 → upsample + add L1  │
│         ↓                               │
│  Reconstructed L1 → upsample + add L0  │
│         ↓                               │
│  Final Reconstructed Image              │
│  (vs. Original comparison)              │
└─────────────────────────────────────────┘
```

**구현**:
```python
def visualize_reconstruction_process(laplacian_pyr, output_path):
    """
    각 reconstruction 단계 시각화
    - 5개 subplot: 각 reconstruction 단계
    - 마지막 subplot: 원본 vs 재구성된 이미지 비교
    """
```

### **Option 2: 에너지/정보량 시각화**
```
바 그래프: 각 레벨별 평균 픽셀값, 표준편차, 에너지량
- Gaussian: 각 레벨의 평균 명도
- Laplacian: 각 레벨의 detail 강도
```

**구현**:
```python
def visualize_pyramid_statistics(gaussian_pyr, laplacian_pyr, output_path):
    """
    Pyramid 레벨별 통계 시각화
    - 막대그래프: 각 레벨의 평균, 표준편차
    - 라인 그래프: 정보량 감소 추이
    """
```

---

## 🎯 최종 요구사항 정리

### **메인 시각화: `pyramid_detailed_layout.png`**
- ✅ Gaussian Pyramid (좌측, 6개 레벨)
- ✅ Laplacian Pyramid - Raw (중앙, 6개 레벨, colormap 적용)
- ✅ Laplacian Pyramid - Brightened (우측, 6개 레벨, 정규화)
- ✅ 각 레벨별 라벨: 레벨 번호, 크기, 설명
- ✅ 명확한 분리선 및 제목
- ✅ 고해상도 (DPI 150+)
- ✅ 저장 경로: `output/visualization/pyramid_detailed_layout.png`

### **추가 시각화 (선택)**
- [ ] Reconstruction 과정 시각화
- [ ] Pyramid 통계 정보 (에너지, 정보량)
- [ ] 각 Laplacian 레벨의 히스토그램

---

## 💾 코드 통합 위치

**`visualization.py`에 다음 함수 추가**:

```python
# 기존 함수들 유지
def visualize_pyramid_levels(pyramid_dict):
    pass

def visualize_blending_comparison(direct, pyramid3, pyramid5, pyramid6, lab):
    pass

# 새로운 함수 추가
def visualize_pyramid_detailed_layout(hand_gaussian_pyr, hand_laplacian_pyr, 
                                      output_path="visualization/pyramid_detailed.png"):
    """
    [위의 상세 구현 내용]
    """
    pass

# 선택: 추가 시각화
def visualize_reconstruction_process(laplacian_pyr, output_path):
    """Reconstruction 과정 시각화"""
    pass

def visualize_pyramid_statistics(gaussian_pyr, laplacian_pyr, output_path):
    """Pyramid 통계 시각화"""
    pass
```

---

## 🔧 `main.py`에서 호출

```python
def main():
    # ... 기존 코드 ...
    
    # 새로운 시각화 추가
    print("[Phase 8] Detailed Pyramid Visualization")
    visualize_pyramid_detailed_layout(
        hand_gaussian_pyr=hand_gp_opencv,  # OpenCV 기반 pyramid
        hand_laplacian_pyr=hand_lp,
        output_path=os.path.join(output_dir, "visualization/pyramid_detailed_layout.png")
    )
    print("✓ Detailed pyramid visualization saved")
    
    # 선택: 추가 시각화
    visualize_reconstruction_process(hand_lp, os.path.join(output_dir, "visualization/reconstruction_process.png"))
    visualize_pyramid_statistics(hand_gp_opencv, hand_lp, os.path.join(output_dir, "visualization/pyramid_statistics.png"))
```

---

## 📌 프로세싱 로그에 추가

```
[Phase 8] Detailed Pyramid Visualization
  ✓ Gaussian Pyramid levels: 6 (480×640 → 15×20)
  ✓ Laplacian Pyramid levels: 6 (480×640 → 15×20)
  ✓ Colormap: JET (for detail visualization)
  ✓ Laplacian normalization: Min-Max [0, 255]
  ✓ Output: visualization/pyramid_detailed_layout.png (High-res PNG)
  ✓ Figure size: (24, 16) at 150 DPI
```

---

## 🎨 최종 결과물 예상

```
output/visualization/pyramid_detailed_layout.png
├── 제목: "Gaussian Pyramid & Laplacian Pyramid Analysis"
├── 좌측: 6개 Gaussian Pyramid 레벨 (원본 컬러)
├── 중앙: 6개 Laplacian Pyramid 레벨 (JET colormap)
├── 우측: 6개 Laplacian Pyramid 레벨 (정규화된 grayscale)
└── 각 이미지: 명확한 라벨, 크기, 설명 포함
```

**이 이미지가 PDF 리포트의 "핵심 시각화"로 사용될 것입니다!** 📄

---

## 추가 질문?

이 시각화가 다음을 포함하도록 설계되었습니다:
1. ✅ 첨부 이미지의 "pyramid 형태" 재현
2. ✅ Gaussian과 Laplacian 동시 비교
3. ✅ 각 레벨의 구조적 변화 명시
4. ✅ PPT 강의 내용 시각화 매핑
5. ✅ "감동을 주는 process" 표현

더 수정이 필요하거나 추가 시각화 아이디어 있으면 알려주세요! 🚀
