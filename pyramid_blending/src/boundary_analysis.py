"""
Boundary Quality Analysis Module

경계 영역의 smoothness를 정량적으로 분석하고 시각화
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import sobel
from matplotlib.patches import Rectangle


def extract_boundary_profile(image, mask, axis='horizontal', position=0.5):
    """
    Extract intensity profile across boundary

    Args:
        image: Image (H, W, 3) or (H, W) in [0, 1]
        mask: Mask (H, W) in [0, 1]
        axis: 'horizontal' or 'vertical'
        position: Position along other axis (0-1)

    Returns:
        profile: Intensity profile
        mask_profile: Mask profile
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    # Ensure mask is 2D
    mask_2d = mask if len(mask.shape) == 2 else mask[:, :, 0]

    h, w = image_gray.shape

    if axis == 'horizontal':
        # Extract row at position
        row_idx = int(h * position)
        profile = image_gray[row_idx, :]
        mask_profile = mask_2d[row_idx, :]
    else:  # vertical
        # Extract column at position
        col_idx = int(w * position)
        profile = image_gray[:, col_idx]
        mask_profile = mask_2d[:, col_idx]

    return profile, mask_profile


def calculate_boundary_gradient_stats(image, mask, threshold_low=0.2, threshold_high=0.8):
    """
    Calculate gradient statistics in boundary region

    Args:
        image: Image (H, W, 3) in [0, 1]
        mask: Mask (H, W) in [0, 1]
        threshold_low: Lower mask threshold for boundary
        threshold_high: Upper mask threshold for boundary

    Returns:
        stats: Dictionary of gradient statistics
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    # Ensure mask is 2D
    mask_2d = mask if len(mask.shape) == 2 else mask[:, :, 0]

    # Define boundary region
    boundary_mask = (mask_2d > threshold_low) & (mask_2d < threshold_high)

    # Calculate gradients
    gradient = sobel(image_gray)

    # Extract boundary gradients
    boundary_gradient = gradient[boundary_mask]

    stats = {
        'mean': float(np.mean(boundary_gradient)),
        'std': float(np.std(boundary_gradient)),
        'max': float(np.max(boundary_gradient)),
        'min': float(np.min(boundary_gradient)),
        'median': float(np.median(boundary_gradient)),
        'num_pixels': int(np.sum(boundary_mask))
    }

    return stats, gradient, boundary_mask


def visualize_boundary_quality(methods_dict, mask, output_path):
    """
    Create comprehensive boundary quality visualization

    Args:
        methods_dict: Dictionary of method_name -> image
        mask: Blending mask
        output_path: Path to save visualization
    """
    # Calculate gradient stats for all methods
    all_stats = {}
    all_gradients = {}
    boundary_mask = None

    for method_name, image in methods_dict.items():
        stats, gradient, b_mask = calculate_boundary_gradient_stats(image, mask)
        all_stats[method_name] = stats
        all_gradients[method_name] = gradient
        if boundary_mask is None:
            boundary_mask = b_mask

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # --- Row 1: Gradient maps for selected methods ---
    methods_to_show = ['direct', 'pyramid_0level', 'pyramid_3level', 'pyramid_5level']
    for idx, method in enumerate(methods_to_show):
        if method not in all_gradients:
            continue

        ax = fig.add_subplot(gs[0, idx])
        gradient = all_gradients[method]

        im = ax.imshow(gradient, cmap='hot', vmin=0, vmax=0.3)
        ax.set_title(f'{method.replace("_", " ").title()}\nGradient Map', fontweight='bold')
        ax.axis('off')

        # Highlight boundary region
        y_coords, x_coords = np.where(boundary_mask)
        if len(y_coords) > 0:
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
            ax.add_patch(rect)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Row 2: Gradient statistics comparison ---
    ax = fig.add_subplot(gs[1, :2])

    # Bar chart of gradient std
    method_names = list(all_stats.keys())
    gradient_stds = [all_stats[m]['std'] for m in method_names]
    gradient_means = [all_stats[m]['mean'] for m in method_names]

    x_pos = np.arange(len(method_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, gradient_means, width, label='Mean', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x_pos + width/2, gradient_stds, width, label='Std Dev', alpha=0.8, color='salmon')

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Gradient Magnitude', fontweight='bold')
    ax.set_title('Boundary Gradient Statistics\n(Lower = Smoother Transition)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=0, ha='center', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8)

    # --- Row 2 Col 3-4: Intensity profiles ---
    ax = fig.add_subplot(gs[1, 2:])

    # Extract horizontal profile at center
    center_y = 0.65  # Eye center area

    for method in methods_to_show:
        if method not in methods_dict:
            continue

        image = methods_dict[method]
        profile, mask_profile = extract_boundary_profile(image, mask, 'horizontal', center_y)

        x_coords = np.arange(len(profile))
        ax.plot(x_coords, profile, label=method.replace('_', ' ').title(), linewidth=2, alpha=0.8)

    # Highlight boundary region on profile
    mask_2d = mask if len(mask.shape) == 2 else mask[:, :, 0]
    row_idx = int(mask_2d.shape[0] * center_y)
    mask_row = mask_2d[row_idx, :]
    boundary_region = (mask_row > 0.2) & (mask_row < 0.8)

    ax.axvspan(np.where(boundary_region)[0][0] if np.any(boundary_region) else 0,
               np.where(boundary_region)[0][-1] if np.any(boundary_region) else len(profile),
               alpha=0.2, color='yellow', label='Boundary Region')

    ax.set_xlabel('Pixel Position (horizontal)', fontweight='bold')
    ax.set_ylabel('Intensity', fontweight='bold')
    ax.set_title(f'Intensity Profile (Row at y={center_y:.2f})\nSmooth = Good Blending', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # --- Row 3: Statistics table ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Create table data
    table_data = [['Method', 'Mean Grad', 'Std Grad', 'Max Grad', 'Median Grad', 'Boundary Pixels']]
    for method in method_names:
        stats = all_stats[method]
        table_data.append([
            method.replace('_', ' '),
            f"{stats['mean']:.6f}",
            f"{stats['std']:.6f}",
            f"{stats['max']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['num_pixels']}"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Highlight best values
    # Find method with lowest std (best)
    best_idx = gradient_stds.index(min(gradient_stds))
    for i in range(len(table_data[0])):
        cell = table[(best_idx + 1, i)]
        cell.set_facecolor('#E8F5E9')

    plt.suptitle('Boundary Quality Analysis: Gradient-based Smoothness Evaluation',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Boundary quality visualization saved: {output_path}")

    return all_stats


def create_boundary_analysis_report(methods_dict, mask, output_dir):
    """
    Create comprehensive boundary analysis

    Args:
        methods_dict: Dictionary of method_name -> image
        mask: Blending mask
        output_dir: Output directory
    """
    import os
    import json

    boundary_dir = os.path.join(output_dir, 'boundary_analysis')
    os.makedirs(boundary_dir, exist_ok=True)

    print("\n[Boundary Analysis] Analyzing boundary quality...")

    # Create visualization
    viz_path = os.path.join(boundary_dir, 'boundary_quality_analysis.png')
    all_stats = visualize_boundary_quality(methods_dict, mask, viz_path)

    # Save statistics to JSON
    stats_path = os.path.join(boundary_dir, 'boundary_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"  ✓ Statistics saved: {stats_path}")

    # Print summary
    print("\n  Boundary Quality Summary:")
    print("  " + "-"*60)
    sorted_methods = sorted(all_stats.items(), key=lambda x: x[1]['std'])
    for rank, (method, stats) in enumerate(sorted_methods, 1):
        print(f"  #{rank} {method:<20} Gradient Std: {stats['std']:.6f}")
    print("  " + "-"*60)

    print(f"\n✓ Boundary analysis saved to: {boundary_dir}/")
