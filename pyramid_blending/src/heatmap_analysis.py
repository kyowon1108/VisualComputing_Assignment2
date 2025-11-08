"""
ROI Heatmap Analysis Module

각 ROI별 품질 메트릭을 히트맵으로 시각화
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim


def calculate_local_ssim(img1, img2, window_size=11):
    """
    Calculate local SSIM at each pixel

    Args:
        img1: First image (H, W, 3) in [0, 1]
        img2: Second image (H, W, 3) in [0, 1]
        window_size: Window size for SSIM calculation

    Returns:
        ssim_map: SSIM value at each pixel (H, W)
    """
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = (img1 * 255).astype(np.uint8)
        img2_gray = (img2 * 255).astype(np.uint8)

    # Calculate SSIM with full output
    ssim_value, ssim_map = ssim(img1_gray, img2_gray,
                                 full=True,
                                 data_range=255,
                                 win_size=window_size)

    return ssim_map


def calculate_local_mse(img1, img2, window_size=11):
    """
    Calculate local MSE using sliding window

    Args:
        img1: First image (H, W, 3) in [0, 1]
        img2: Second image (H, W, 3) in [0, 1]
        window_size: Window size for MSE calculation

    Returns:
        mse_map: MSE value at each pixel (H, W)
    """
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)

    h, w = img1.shape
    mse_map = np.zeros((h, w), dtype=np.float32)

    # Calculate squared difference
    sq_diff = (img1 - img2) ** 2

    # Apply box filter to get local MSE
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    mse_map = cv2.filter2D(sq_diff, -1, kernel)

    return mse_map


def create_roi_heatmap(result_img, reference_img, mask, output_path, method_name='Pyramid'):
    """
    Create ROI-based quality heatmaps

    Args:
        result_img: Result image (H, W, 3) in [0, 1]
        reference_img: Reference image (H, W, 3) in [0, 1]
        mask: Blending mask (H, W) in [0, 1]
        output_path: Path to save heatmap
        method_name: Name of the blending method
    """
    # Calculate local metrics
    ssim_map = calculate_local_ssim(result_img, reference_img)
    mse_map = calculate_local_mse(result_img, reference_img)

    # Define ROI regions (same as roi_analysis.py)
    h, w = mask.shape[:2] if len(mask.shape) == 3 else mask.shape
    center = (325, 315)
    axes = (48, 36)

    # Ensure mask is 2D
    mask_2d = mask if len(mask.shape) == 2 else mask[:, :, 0]

    # Calculate ellipse distance
    y_coords, x_coords = np.ogrid[:h, :w]
    dx = (x_coords - center[1]) / axes[0]
    dy = (y_coords - center[0]) / axes[1]
    ellipse_dist = np.sqrt(dx**2 + dy**2)

    # Define ROI masks
    roi_hand = np.zeros((h, w), dtype=bool)
    finger_x_start, finger_x_end = 370, 430
    finger_y_start, finger_y_end = 80, 180
    roi_hand[finger_y_start:finger_y_end, finger_x_start:finger_x_end] = True
    roi_hand = roi_hand & (ellipse_dist > 3.0)

    roi_eye = ellipse_dist < 1.2
    roi_boundary = (ellipse_dist >= 1.2) & (ellipse_dist < 2.5)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: SSIM
    # Col 1: Result image with ROI overlay
    ax = axes[0, 0]
    ax.imshow(result_img)
    ax.set_title(f'{method_name} Blending Result\nwith ROI Overlay', fontsize=12, fontweight='bold')

    # Draw ROI boxes
    # Hand ROI (red)
    rect_hand = plt.Rectangle((finger_x_start, finger_y_start),
                              finger_x_end - finger_x_start,
                              finger_y_end - finger_y_start,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect_hand)
    ax.text(finger_x_start, finger_y_start - 5, 'Hand',
            color='red', fontsize=10, fontweight='bold')

    # Eye ROI (blue)
    y_min_eye, y_max_eye = np.where(roi_eye.any(axis=1))[0][[0, -1]]
    x_min_eye, x_max_eye = np.where(roi_eye.any(axis=0))[0][[0, -1]]
    rect_eye = plt.Rectangle((x_min_eye, y_min_eye),
                             x_max_eye - x_min_eye,
                             y_max_eye - y_min_eye,
                             fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect_eye)
    ax.text(x_min_eye, y_min_eye - 5, 'Eye',
            color='blue', fontsize=10, fontweight='bold')

    # Boundary ROI (green)
    y_min_bd, y_max_bd = np.where(roi_boundary.any(axis=1))[0][[0, -1]]
    x_min_bd, x_max_bd = np.where(roi_boundary.any(axis=0))[0][[0, -1]]
    rect_bd = plt.Rectangle((x_min_bd, y_min_bd),
                            x_max_bd - x_min_bd,
                            y_max_bd - y_min_bd,
                            fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect_bd)
    ax.text(x_min_bd, y_min_bd - 5, 'Boundary',
            color='green', fontsize=10, fontweight='bold')

    ax.axis('off')

    # Col 2: SSIM heatmap
    ax = axes[0, 1]
    im_ssim = ax.imshow(ssim_map, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_title('SSIM Heatmap\n(Red=Low, Green=High)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im_ssim, ax=ax, fraction=0.046, pad=0.04)

    # Col 3: SSIM statistics per ROI
    ax = axes[0, 2]
    ax.axis('off')

    # Calculate ROI statistics
    ssim_hand = ssim_map[roi_hand]
    ssim_eye = ssim_map[roi_eye]
    ssim_boundary = ssim_map[roi_boundary]

    stats_text = f"SSIM Statistics by ROI\n\n"
    stats_text += f"{'Region':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n"
    stats_text += "-" * 50 + "\n"
    stats_text += f"{'Hand':<12} {ssim_hand.mean():<8.4f} {ssim_hand.std():<8.4f} {ssim_hand.min():<8.4f} {ssim_hand.max():<8.4f}\n"
    stats_text += f"{'Eye':<12} {ssim_eye.mean():<8.4f} {ssim_eye.std():<8.4f} {ssim_eye.min():<8.4f} {ssim_eye.max():<8.4f}\n"
    stats_text += f"{'Boundary':<12} {ssim_boundary.mean():<8.4f} {ssim_boundary.std():<8.4f} {ssim_boundary.min():<8.4f} {ssim_boundary.max():<8.4f}\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: MSE
    # Col 1: MSE heatmap
    ax = axes[1, 1]
    # Use log scale for better visualization
    mse_log = np.log10(mse_map + 1e-10)
    im_mse = ax.imshow(mse_log, cmap='hot_r', vmin=-4, vmax=-1)
    ax.set_title('MSE Heatmap (log scale)\n(Dark=Low, Bright=High)', fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im_mse, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', rotation=270, labelpad=15)

    # Col 2: Difference image
    ax = axes[1, 0]
    diff = np.abs(result_img - reference_img)
    if len(diff.shape) == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff
    im_diff = ax.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.1)
    ax.set_title('Absolute Difference\n(Black=No diff, Red=High diff)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im_diff, ax=ax, fraction=0.046, pad=0.04)

    # Col 3: MSE statistics per ROI
    ax = axes[1, 2]
    ax.axis('off')

    mse_hand = mse_map[roi_hand]
    mse_eye = mse_map[roi_eye]
    mse_boundary = mse_map[roi_boundary]

    stats_text = f"MSE Statistics by ROI\n\n"
    stats_text += f"{'Region':<12} {'Mean':<12} {'Std':<12}\n"
    stats_text += "-" * 40 + "\n"
    stats_text += f"{'Hand':<12} {mse_hand.mean():<12.6f} {mse_hand.std():<12.6f}\n"
    stats_text += f"{'Eye':<12} {mse_eye.mean():<12.6f} {mse_eye.std():<12.6f}\n"
    stats_text += f"{'Boundary':<12} {mse_boundary.mean():<12.6f} {mse_boundary.std():<12.6f}\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'ROI Quality Heatmap Analysis: {method_name} Blending',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ ROI heatmap saved: {output_path}")


def create_all_heatmaps(methods_dict, reference_img, mask, output_dir):
    """
    Create heatmaps for all blending methods

    Args:
        methods_dict: Dictionary of method_name -> result_image
        reference_img: Reference image (direct blending)
        mask: Blending mask
        output_dir: Output directory
    """
    import os
    heatmap_dir = os.path.join(output_dir, 'heatmap_analysis')
    os.makedirs(heatmap_dir, exist_ok=True)

    print("\n[Heatmap Analysis] Creating quality heatmaps...")

    for method_name, result_img in methods_dict.items():
        if method_name == 'direct':
            continue  # Skip reference

        output_path = os.path.join(heatmap_dir, f'heatmap_{method_name}.png')
        create_roi_heatmap(result_img, reference_img, mask, output_path, method_name)

    print(f"\n✓ All heatmaps saved to: {heatmap_dir}/")
