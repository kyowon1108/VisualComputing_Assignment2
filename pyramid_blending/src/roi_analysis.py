"""
ROI (Region of Interest) Analysis Module
Hand, Eye, Transition 영역별 시각적/정량적 평가
"""
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim


def extract_roi(image, mask, roi_type='hand'):
    """
    ROI 추출 (특정 영역 기반 정의)

    Args:
        image: 입력 이미지 (H, W, 3)
        mask: 마스크 (H, W, 1) 또는 (H, W)
        roi_type: 'hand', 'eye', 'boundary' 중 하나

    Returns:
        roi_mask: Boolean mask for ROI
        bbox: Bounding box (x, y, w, h)
    """
    # Squeeze mask to 2D
    if len(mask.shape) == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    h, w = mask_2d.shape

    # Define ellipse parameters (from preprocessing)
    center = (325, 315)  # (row, col)
    axes = (48, 36)      # (axis_x, axis_y)

    # Create distance map from ellipse center
    y_coords, x_coords = np.ogrid[:h, :w]

    # Normalized distance from ellipse center
    dx = (x_coords - center[1]) / axes[0]  # center[1] = col = 315
    dy = (y_coords - center[0]) / axes[1]  # center[0] = row = 325
    ellipse_dist = np.sqrt(dx**2 + dy**2)

    # Define ROI based on specific regions
    if roi_type == 'hand':
        # Hand region: 검지손가락 영역
        # 손가락은 대략 상단 중앙 위치 (이미지 기준)
        # 검지손가락 대략 위치: x=370~430, y=80~180
        roi_mask = np.zeros((h, w), dtype=bool)
        finger_x_start, finger_x_end = 370, 430
        finger_y_start, finger_y_end = 80, 180
        roi_mask[finger_y_start:finger_y_end, finger_x_start:finger_x_end] = True

        # 타원 영역은 제외 (눈이 없는 순수 손가락 부분만)
        roi_mask = roi_mask & (ellipse_dist > 3.0)

    elif roi_type == 'eye':
        # Eye region: 눈 전체 영역 (타원 내부)
        # 눈이 실제로 위치한 영역만 포함
        roi_mask = ellipse_dist < 1.2

    elif roi_type == 'boundary':
        # Boundary region: 눈 경계 링 (블렌딩 전환 영역)
        # 눈 바로 주변의 링 모양 영역
        roi_mask = (ellipse_dist >= 1.2) & (ellipse_dist < 2.5)

    else:
        raise ValueError(f"Unknown ROI type: {roi_type}")

    # Find bounding box
    rows, cols = np.where(roi_mask)
    if len(rows) == 0:
        return roi_mask, None

    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()

    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    return roi_mask, bbox


def calculate_roi_metrics(result, reference, mask, roi_type='hand'):
    """
    ROI별 정량적 메트릭 계산

    Args:
        result: 블렌딩 결과 이미지
        reference: 참조 이미지 (Direct blending)
        mask: 마스크
        roi_type: ROI 타입

    Returns:
        metrics: Dictionary of metrics
    """
    roi_mask, bbox = extract_roi(result, mask, roi_type)

    if bbox is None or roi_mask.sum() == 0:
        return None

    metrics = {}

    # 1. SSIM (Structural Similarity)
    roi_result = result[roi_mask]
    roi_reference = reference[roi_mask]

    # Expand to 2D for SSIM calculation
    if len(roi_result.shape) == 1:
        # Need to reconstruct 2D patch
        y_min, y_max = np.where(roi_mask)[0].min(), np.where(roi_mask)[0].max()
        x_min, x_max = np.where(roi_mask)[1].min(), np.where(roi_mask)[1].max()

        result_patch = result[y_min:y_max+1, x_min:x_max+1]
        ref_patch = reference[y_min:y_max+1, x_min:x_max+1]
        roi_mask_patch = roi_mask[y_min:y_max+1, x_min:x_max+1]

        # Calculate SSIM on patch
        if result_patch.size > 49:  # Minimum size for SSIM
            ssim_value = ssim(result_patch, ref_patch,
                            data_range=1.0,
                            channel_axis=2 if len(result_patch.shape) == 3 else None)
            metrics['ssim'] = float(ssim_value)
        else:
            metrics['ssim'] = None

    # 2. MSE (Mean Squared Error)
    mse = np.mean((roi_result - roi_reference) ** 2)
    metrics['mse'] = float(mse)

    # 3. Mean intensity
    metrics['mean_intensity'] = float(np.mean(roi_result))

    # 4. Std intensity
    metrics['std_intensity'] = float(np.std(roi_result))

    # 5. ROI-specific metrics
    if roi_type == 'hand':
        # Hand region: texture preservation
        # Calculate variance as texture indicator
        metrics['texture_variance'] = float(np.var(roi_result))

    elif roi_type == 'eye':
        # Eye region: contrast
        # Use patch instead of masked pixels
        y_min, y_max = np.where(roi_mask)[0].min(), np.where(roi_mask)[0].max()
        x_min, x_max = np.where(roi_mask)[1].min(), np.where(roi_mask)[1].max()

        eye_patch = result[y_min:y_max+1, x_min:x_max+1]

        if len(eye_patch.shape) == 3:
            gray_result = cv2.cvtColor((eye_patch * 255).astype(np.uint8),
                                      cv2.COLOR_RGB2GRAY)
            roi_mask_patch = roi_mask[y_min:y_max+1, x_min:x_max+1]
            gray_roi = gray_result[roi_mask_patch]
            metrics['contrast'] = float(gray_roi.max() - gray_roi.min()) / 255.0

    elif roi_type == 'boundary':
        # Boundary region: smoothness (gradient analysis)
        y_min, y_max = np.where(roi_mask)[0].min(), np.where(roi_mask)[0].max()
        x_min, x_max = np.where(roi_mask)[1].min(), np.where(roi_mask)[1].max()

        transition_patch = result[y_min:y_max+1, x_min:x_max+1]

        # Convert to grayscale
        if len(transition_patch.shape) == 3:
            gray = cv2.cvtColor((transition_patch * 255).astype(np.uint8),
                               cv2.COLOR_RGB2GRAY)
        else:
            gray = (transition_patch * 255).astype(np.uint8)

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Gradient statistics in transition region
        transition_mask_patch = roi_mask[y_min:y_max+1, x_min:x_max+1]
        transition_grad = gradient[transition_mask_patch]

        metrics['gradient_mean'] = float(np.mean(transition_grad))
        metrics['gradient_std'] = float(np.std(transition_grad))
        metrics['gradient_max'] = float(np.max(transition_grad))

    # 6. Pixel count and coverage
    metrics['pixel_count'] = int(roi_mask.sum())
    metrics['coverage_percent'] = float(roi_mask.sum() / roi_mask.size * 100)

    # 7. Bounding box (convert to native Python types for JSON)
    metrics['bbox'] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    return metrics


def visualize_roi_locations(image, mask, output_path):
    """
    ROI 위치를 빨간 박스로 표시

    Args:
        image: 원본 이미지
        mask: 마스크
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Display image
    ax.imshow(image)
    ax.set_title('ROI Locations', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Define ROI types and colors
    roi_configs = [
        ('hand', 'red', 'ROI-1: Hand (Index Finger)'),
        ('eye', 'blue', 'ROI-2: Eye (Center)'),
        ('boundary', 'green', 'ROI-3: Boundary (Ring)')
    ]

    for roi_type, color, label in roi_configs:
        roi_mask, bbox = extract_roi(image, mask, roi_type)

        if bbox is not None:
            x, y, w, h = bbox

            # Draw rectangle
            rect = Rectangle((x, y), w, h,
                           linewidth=3,
                           edgecolor=color,
                           facecolor='none',
                           linestyle='-')
            ax.add_patch(rect)

            # Add label
            ax.text(x, y - 10, label,
                   color=color,
                   fontsize=12,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           edgecolor=color,
                           alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ ROI 위치 시각화 저장: {output_path}")


def visualize_roi_comparison(result, reference, mask, output_path, original_hand=None, original_eye=None):
    """
    ROI comparison: Original | Direct Blend | Pyramid Blend

    Args:
        result: Pyramid blending result
        reference: Direct blending result
        mask: Mask
        output_path: Save path
        original_hand: Original hand image
        original_eye: Original eye image
    """
    roi_types = ['hand', 'eye', 'boundary']
    roi_names = ['Hand Region', 'Eye Region', 'Boundary Region']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for idx, (roi_type, roi_name) in enumerate(zip(roi_types, roi_names)):
        roi_mask, bbox = extract_roi(result, mask, roi_type)

        if bbox is None:
            continue

        x, y, w, h = bbox

        # Extract patches
        direct_patch = reference[y:y+h, x:x+w]  # Direct blend
        pyramid_patch = result[y:y+h, x:x+w]    # Pyramid blend

        # Original patch (depends on ROI type)
        if roi_type == 'hand' and original_hand is not None:
            orig_patch = original_hand[y:y+h, x:x+w]
        elif roi_type == 'eye' and original_eye is not None:
            orig_patch = original_eye[y:y+h, x:x+w]
        elif roi_type == 'boundary' and original_hand is not None:
            orig_patch = original_hand[y:y+h, x:x+w]
        else:
            orig_patch = direct_patch  # Fallback

        # Column 1: Original
        axes[idx, 0].imshow(orig_patch)
        axes[idx, 0].set_title(f'{roi_name}\n(Original)', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        # Column 2: Direct Blend
        axes[idx, 1].imshow(direct_patch)
        axes[idx, 1].set_title(f'{roi_name}\n(Direct Blend)', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

        # Column 3: Pyramid Blend
        axes[idx, 2].imshow(pyramid_patch)
        axes[idx, 2].set_title(f'{roi_name}\n(Pyramid Blend)', fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.suptitle('ROI Comparison: Original vs Direct Blend vs Pyramid Blend',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ROI comparison saved: {output_path}")


def generate_roi_report(all_metrics, output_path):
    """
    ROI 분석 텍스트 리포트 생성

    Args:
        all_metrics: Dictionary of all ROI metrics
        output_path: 저장 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ROI (Region of Interest) Analysis Report\n")
        f.write("="*80 + "\n\n")

        # ROI-1: Hand Region
        f.write("ROI-1: Hand Region (검지손가락 영역)\n")
        f.write("-"*80 + "\n")
        if 'hand' in all_metrics and all_metrics['hand']:
            m = all_metrics['hand']
            f.write(f"  픽셀 수: {m['pixel_count']:,} pixels ({m['coverage_percent']:.1f}%)\n")
            if m.get('ssim') is not None:
                f.write(f"  SSIM: {m['ssim']:.4f}\n")
            f.write(f"  MSE: {m['mse']:.6f}\n")
            f.write(f"  평균 밝기: {m['mean_intensity']:.4f}\n")
            f.write(f"  밝기 표준편차: {m['std_intensity']:.4f}\n")
            if 'texture_variance' in m:
                f.write(f"  텍스처 분산: {m['texture_variance']:.6f}\n")
            bbox = m['bbox']
            f.write(f"  Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}\n")
            f.write("\n  평가: Hand 영역의 텍스처와 색상이 원본 대비 잘 보존되었는지 확인\n")
            f.write("       이 영역은 블렌딩의 영향을 받지 않아야 함 (순수 손가락 영역)\n")
        f.write("\n")

        # ROI-2: Eye Region
        f.write("ROI-2: Eye Region (눈 전체, distance < 1.2)\n")
        f.write("-"*80 + "\n")
        if 'eye' in all_metrics and all_metrics['eye']:
            m = all_metrics['eye']
            f.write(f"  픽셀 수: {m['pixel_count']:,} pixels ({m['coverage_percent']:.1f}%)\n")
            if m.get('ssim') is not None:
                f.write(f"  SSIM: {m['ssim']:.4f}\n")
            f.write(f"  MSE: {m['mse']:.6f}\n")
            f.write(f"  평균 밝기: {m['mean_intensity']:.4f}\n")
            f.write(f"  밝기 표준편차: {m['std_intensity']:.4f}\n")
            if 'contrast' in m:
                f.write(f"  대비(Contrast): {m['contrast']:.4f}\n")
            bbox = m['bbox']
            f.write(f"  Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}\n")
            f.write("\n  평가: Eye 영역의 디테일과 대비가 명확하게 유지되었는지 확인\n")
            f.write("       이 영역은 Eye 이미지가 나타나는 핵심 영역\n")
        f.write("\n")

        # ROI-3: Boundary Region
        f.write("ROI-3: Boundary Region (눈 경계 링, 1.2 ≤ distance < 2.5)\n")
        f.write("-"*80 + "\n")
        if 'boundary' in all_metrics and all_metrics['boundary']:
            m = all_metrics['boundary']
            f.write(f"  픽셀 수: {m['pixel_count']:,} pixels ({m['coverage_percent']:.1f}%)\n")
            if m.get('ssim') is not None:
                f.write(f"  SSIM: {m['ssim']:.4f}\n")
            f.write(f"  MSE: {m['mse']:.6f}\n")
            f.write(f"  평균 밝기: {m['mean_intensity']:.4f}\n")
            f.write(f"  밝기 표준편차: {m['std_intensity']:.4f}\n")
            if 'gradient_mean' in m:
                f.write(f"  평균 Gradient: {m['gradient_mean']:.4f}\n")
            if 'gradient_std' in m:
                f.write(f"  Gradient 표준편차: {m['gradient_std']:.4f} {'✓ Smooth' if m['gradient_std'] < 50.0 else '✗ Rough'}\n")
            if 'gradient_max' in m:
                f.write(f"  최대 Gradient: {m['gradient_max']:.4f}\n")
            bbox = m['bbox']
            f.write(f"  Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}\n")
            f.write("\n  평가: 경계 영역이 부드럽게 블렌딩되었는지 확인\n")
            f.write("       → Pyramid Blending의 핵심 장점을 보여주는 영역\n")
            f.write("       → Gradient Std가 낮을수록 부드러운 전환 (목표: < 50.0)\n")
        f.write("\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("종합 평가\n")
        f.write("="*80 + "\n")
        f.write("✓ ROI-1 (Hand): 원본 텍스처 보존 여부\n")
        f.write("✓ ROI-2 (Eye): 눈 디테일 유지 여부\n")
        f.write("✓ ROI-3 (Transition): 부드러운 전환 여부 ← 가장 중요!\n")
        f.write("\n")
        f.write("Pyramid Blending의 성공 여부는 ROI-3의 Gradient Std가 낮을수록 좋음\n")
        f.write("Direct Blending 대비 ROI-3에서 현저히 낮은 Gradient Std를 보이면 성공\n")

    print(f"  ✓ ROI 리포트 저장: {output_path}")


def analyze_all_methods(methods_dict, mask, output_dir, original_hand=None, original_eye=None):
    """
    ROI analysis for multiple blending methods

    Args:
        methods_dict: {method_name: image} dictionary
        mask: Mask
        output_dir: Output directory
        original_hand: Original hand image
        original_eye: Original eye image

    Returns:
        all_results: All ROI metrics
    """
    print("\n" + "="*60)
    print("ROI Analysis for All Blending Methods")
    print("="*60)

    # Create output directory
    roi_dir = os.path.join(output_dir, 'roi_analysis')
    os.makedirs(roi_dir, exist_ok=True)

    # Reference: Direct blending
    if 'direct' not in methods_dict:
        print("Warning: No direct blending found, using first method as reference")
        reference = list(methods_dict.values())[0]
    else:
        reference = methods_dict['direct']

    all_results = {}

    # Analyze each method
    for method_name, result_image in methods_dict.items():
        print(f"\n분석 중: {method_name}")

        method_metrics = {}

        for roi_type in ['hand', 'eye', 'boundary']:
            metrics = calculate_roi_metrics(result_image, reference, mask, roi_type)
            method_metrics[roi_type] = metrics

            if metrics:
                print(f"  ✓ {roi_type.upper()}: {metrics['pixel_count']} pixels")

        all_results[method_name] = method_metrics

    # Save metrics to JSON
    metrics_path = os.path.join(roi_dir, 'roi_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ ROI 메트릭 저장: {metrics_path}")

    # Visualize ROI locations (using first image)
    first_image = list(methods_dict.values())[0]
    viz_path = os.path.join(roi_dir, 'roi_locations.png')
    visualize_roi_locations(first_image, mask, viz_path)

    # Visualize comparison (Direct vs Pyramid Level 0)
    if 'direct' in methods_dict and 'pyramid_0level' in methods_dict:
        comp_path = os.path.join(roi_dir, 'roi_comparison.png')
        visualize_roi_comparison(methods_dict['pyramid_0level'],
                                methods_dict['direct'],
                                mask,
                                comp_path,
                                original_hand,
                                original_eye)

    # Generate text report (for best method)
    if 'pyramid_0level' in all_results:
        report_path = os.path.join(roi_dir, 'roi_report.txt')
        generate_roi_report(all_results['pyramid_0level'], report_path)

    print("\n" + "="*60)
    print("✓ ROI Analysis Complete!")
    print("="*60)
    print(f"\nOutput directory: {roi_dir}")
    print("  - roi_metrics.json: 정량적 메트릭")
    print("  - roi_locations.png: ROI 위치 시각화")
    print("  - roi_comparison.png: ROI 확대 비교")
    print("  - roi_report.txt: 텍스트 리포트")

    return all_results


def create_roi_summary_table(all_results, output_path):
    """
    ROI 메트릭 비교표 생성 (이미지)

    Args:
        all_results: 모든 방법의 ROI 메트릭
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ['Method', 'ROI-1 (Hand)\nSSIM / MSE', 'ROI-2 (Eye)\nSSIM / Contrast',
               'ROI-3 (Boundary)\nGrad Std / Status']

    table_data = []
    for method_name, metrics in all_results.items():
        row = [method_name]

        # ROI-1
        if metrics.get('hand'):
            ssim_val = metrics['hand'].get('ssim', 'N/A')
            mse_val = metrics['hand'].get('mse', 'N/A')
            if isinstance(ssim_val, float):
                row.append(f"{ssim_val:.4f} / {mse_val:.6f}")
            else:
                row.append('N/A')
        else:
            row.append('N/A')

        # ROI-2
        if metrics.get('eye'):
            ssim_val = metrics['eye'].get('ssim', 'N/A')
            contrast = metrics['eye'].get('contrast', 'N/A')
            if isinstance(ssim_val, float) and isinstance(contrast, float):
                row.append(f"{ssim_val:.4f} / {contrast:.4f}")
            else:
                row.append('N/A')
        else:
            row.append('N/A')

        # ROI-3
        if metrics.get('boundary'):
            grad_std = metrics['boundary'].get('gradient_std', 'N/A')
            if isinstance(grad_std, float):
                status = '✓ Smooth' if grad_std < 0.05 else '○ OK' if grad_std < 0.10 else '✗ Rough'
                row.append(f"{grad_std:.4f} / {status}")
            else:
                row.append('N/A')
        else:
            row.append('N/A')

        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.27, 0.27, 0.26])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('ROI Analysis Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ ROI 요약 테이블 저장: {output_path}")
