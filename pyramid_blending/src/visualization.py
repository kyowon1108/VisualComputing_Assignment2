"""
Visualization module for results and analysis
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os


def visualize_pyramid_levels(pyramid_dict, output_path):
    """
    Visualize all pyramid levels

    Args:
        pyramid_dict: Dictionary with keys 'hand_gp', 'eye_gp', 'mask_gp'
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Number of pyramid types and levels
    pyramid_names = list(pyramid_dict.keys())
    n_pyramids = len(pyramid_names)
    n_levels = len(pyramid_dict[pyramid_names[0]])

    # Create figure
    fig, axes = plt.subplots(n_pyramids, n_levels, figsize=(20, 6))

    if n_pyramids == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(pyramid_names):
        pyramid = pyramid_dict[name]

        for j, level in enumerate(pyramid):
            ax = axes[i, j]

            # Normalize for display
            if level.dtype != np.uint8:
                display_img = np.clip(level, 0, 1)
            else:
                display_img = level

            # Handle grayscale
            if len(display_img.shape) == 2:
                ax.imshow(display_img, cmap='gray')
            elif display_img.shape[2] == 1:
                ax.imshow(display_img[:, :, 0], cmap='gray')
            else:
                ax.imshow(display_img)

            ax.set_title(f"{name}\nLevel {j}: {level.shape[:2]}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Pyramid comparison saved to: {output_path}")


def visualize_blending_comparison(results_dict, metrics_dict, output_path):
    """
    Visualize blending method comparison

    Args:
        results_dict: Dictionary mapping method names to result images
        metrics_dict: Dictionary mapping method names to metrics
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(20, 5))

    if n_methods == 1:
        axes = [axes]

    for i, (method, image) in enumerate(results_dict.items()):
        # Normalize for display
        if image.dtype != np.uint8:
            display_img = np.clip(image, 0, 1)
        else:
            display_img = image / 255.0

        axes[i].imshow(display_img)

        # Get metrics
        metrics = metrics_dict.get(method, {})
        ssim_val = metrics.get('ssim', 0)
        mse_val = metrics.get('mse', 0)

        axes[i].set_title(f"{method}\nSSIM: {ssim_val:.4f} | MSE: {mse_val:.4f}",
                         fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Blending comparison saved to: {output_path}")


def plot_quality_metrics(metrics_dict, output_path):
    """
    Plot quality metrics as bar charts

    Args:
        metrics_dict: Dictionary of metrics
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    methods = list(metrics_dict.keys())
    ssim_values = [metrics_dict[m]['ssim'] for m in methods]
    mse_values = [metrics_dict[m]['mse'] for m in methods]
    psnr_values = [metrics_dict[m]['psnr'] if metrics_dict[m]['psnr'] != float('inf') else 50
                   for m in methods]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # SSIM plot
    axes[0].bar(range(len(methods)), ssim_values, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('SSIM (higher is better)')
    axes[0].set_title('Structural Similarity Index (SSIM)')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    # MSE plot
    axes[1].bar(range(len(methods)), mse_values, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('MSE (lower is better)')
    axes[1].set_title('Mean Squared Error (MSE)')
    axes[1].grid(axis='y', alpha=0.3)

    # PSNR plot
    axes[2].bar(range(len(methods)), psnr_values, color='lightgreen', alpha=0.7)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[2].set_ylabel('PSNR (higher is better)')
    axes[2].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Quality metrics plot saved to: {output_path}")


def plot_histogram_comparison(results_dict, output_path):
    """
    Plot histogram comparison

    Args:
        results_dict: Dictionary of result images
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, len(results_dict), figsize=(18, 4))

    if len(results_dict) == 1:
        axes = [axes]

    for i, (method, image) in enumerate(results_dict.items()):
        # Convert to grayscale for histogram
        if len(image.shape) == 3:
            if image.dtype != np.uint8:
                gray = np.mean(image, axis=2) * 255
            else:
                gray = np.mean(image, axis=2)
        else:
            gray = image if image.dtype == np.uint8 else image * 255

        gray = gray.astype(np.uint8)

        # Plot histogram
        axes[i].hist(gray.flatten(), bins=64, color='steelblue', alpha=0.7)
        axes[i].set_title(f"{method}\nIntensity Histogram", fontsize=10)
        axes[i].set_xlabel('Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Histogram comparison saved to: {output_path}")


def visualize_level_comparison(results_dict, metrics_dict, output_path):
    """
    Visualize comparison of different pyramid levels

    Args:
        results_dict: Dictionary mapping level counts to result images
        metrics_dict: Dictionary mapping level counts to metrics
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_levels = len(results_dict)
    fig, axes = plt.subplots(1, n_levels, figsize=(18, 5))

    if n_levels == 1:
        axes = [axes]

    for i, (level, image) in enumerate(sorted(results_dict.items())):
        # Normalize for display
        if image.dtype != np.uint8:
            display_img = np.clip(image, 0, 1)
        else:
            display_img = image / 255.0

        axes[i].imshow(display_img)

        # Get metrics
        metrics = metrics_dict.get(f'pyramid_{level}level', {})
        ssim_val = metrics.get('ssim', 0)
        mse_val = metrics.get('mse', 0)

        axes[i].set_title(f"{level}-Level Pyramid\nSSIM: {ssim_val:.4f} | MSE: {mse_val:.4f}",
                         fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Level comparison saved to: {output_path}")


def create_side_by_side_comparison(image1, image2, title1, title2, output_path):
    """
    Create side-by-side comparison of two images

    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Normalize images
    if image1.dtype != np.uint8:
        image1 = np.clip(image1, 0, 1)
    if image2.dtype != np.uint8:
        image2 = np.clip(image2, 0, 1)

    axes[0].imshow(image1)
    axes[0].set_title(title1, fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title(title2, fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
