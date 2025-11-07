"""
Comparison experiments module
"""
import os
from .blending import pyramid_blending, lab_blending, direct_blending
from .metrics import calculate_metrics
from .utils import save_image


def compare_pyramid_levels(hand_lap, eye_lap, mask_gp, hand_rgb, eye_rgb,
                           output_dir, reference=None):
    """
    Compare different pyramid levels: 3 vs 5 vs 6

    Args:
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        hand_rgb: Hand RGB image (for reference)
        eye_rgb: Eye RGB image (for reference)
        output_dir: Output directory
        reference: Optional reference image for metrics

    Returns:
        results: Dictionary mapping level count to blended image
        metrics_dict: Dictionary of metrics
    """
    print("\n[Comparison] Pyramid Levels (3 vs 5 vs 6)")

    results = {}
    metrics_dict = {}

    levels_to_test = [3, 5, 6]

    for levels in levels_to_test:
        print(f"  Testing {levels}-level pyramid...")

        # Blend using specified number of levels
        blended = pyramid_blending(hand_lap, eye_lap, mask_gp, levels)
        results[levels] = blended

        # Save result
        output_path = os.path.join(output_dir, 'blending_results',
                                  f'pyramid_{levels}level.jpg')
        save_image(blended, output_path)

        # Calculate metrics if reference provided
        if reference is not None:
            metrics = calculate_metrics(blended, reference)
            metrics_dict[f'pyramid_{levels}level'] = metrics
            print(f"    ✓ SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.4f}")

    return results, metrics_dict


def compare_direct_vs_pyramid(hand_img, eye_img, mask, hand_lap, eye_lap,
                              mask_gp, output_dir, reference=None):
    """
    Compare direct blending vs pyramid blending

    Args:
        hand_img: Hand image
        eye_img: Eye image
        mask: Blending mask
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        output_dir: Output directory
        reference: Optional reference image for metrics

    Returns:
        results: Dictionary of results
        metrics_dict: Dictionary of metrics
    """
    print("\n[Comparison] Direct vs Pyramid Blending")

    results = {}
    metrics_dict = {}

    # Direct blending
    print("  Testing direct blending...")
    direct_result = direct_blending(hand_img, eye_img, mask)
    results['direct'] = direct_result

    # Save result
    output_path = os.path.join(output_dir, 'blending_results', 'direct_blend.jpg')
    save_image(direct_result, output_path)

    if reference is not None:
        metrics = calculate_metrics(direct_result, reference)
        metrics_dict['direct_blending'] = metrics
        print(f"    ✓ SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.4f}")

    # Pyramid blending (5-level)
    print("  Testing pyramid blending (5-level)...")
    pyramid_result = pyramid_blending(hand_lap, eye_lap, mask_gp, 5)
    results['pyramid_5'] = pyramid_result

    # Save result
    output_path = os.path.join(output_dir, 'blending_results', 'pyramid_5level.jpg')
    save_image(pyramid_result, output_path)

    if reference is not None:
        metrics = calculate_metrics(pyramid_result, reference)
        metrics_dict['pyramid_5level'] = metrics
        print(f"    ✓ SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.4f}")

    return results, metrics_dict


def compare_color_spaces(hand_lap, eye_lap, mask_gp, hand_rgb, eye_rgb,
                         output_dir, reference=None):
    """
    Compare RGB vs LAB color space blending

    Args:
        hand_lap: Hand Laplacian pyramid (RGB)
        eye_lap: Eye Laplacian pyramid (RGB)
        mask_gp: Mask Gaussian pyramid
        hand_rgb: Hand RGB image
        eye_rgb: Eye RGB image
        output_dir: Output directory
        reference: Optional reference image for metrics

    Returns:
        results: Dictionary of results
        metrics_dict: Dictionary of metrics
    """
    print("\n[Comparison] RGB vs LAB Color Space")

    results = {}
    metrics_dict = {}

    # RGB blending (already done in previous comparison)
    print("  RGB blending (5-level)...")
    rgb_result = pyramid_blending(hand_lap, eye_lap, mask_gp, 5)
    results['rgb'] = rgb_result

    # LAB blending
    print("  LAB blending (5-level)...")
    lab_result = lab_blending(hand_lap, eye_lap, mask_gp, hand_rgb, eye_rgb, 5)
    results['lab'] = lab_result

    # Save result
    output_path = os.path.join(output_dir, 'blending_results', 'lab_blend_5level.jpg')
    save_image(lab_result, output_path)

    if reference is not None:
        metrics = calculate_metrics(lab_result, reference)
        metrics_dict['lab_blend_5level'] = metrics
        print(f"    ✓ SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.4f}")

    return results, metrics_dict


def run_all_comparisons(hand_img, eye_img, mask, hand_lap, eye_lap, mask_gp,
                       output_dir):
    """
    Run all comparison experiments

    Args:
        hand_img: Hand image
        eye_img: Eye image
        mask: Blending mask
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        output_dir: Output directory

    Returns:
        all_results: Dictionary of all results
        all_metrics: Dictionary of all metrics
    """
    all_results = {}
    all_metrics = {}

    # First, create direct blending (baseline/reference)
    direct_result = direct_blending(hand_img, eye_img, mask)

    print(f"\n[Debug] Reference shape: {direct_result.shape}, dtype: {direct_result.dtype}")

    # Use direct blending as reference for comparison
    reference = direct_result

    # Compare pyramid levels (using direct blend as reference)
    level_results, level_metrics = compare_pyramid_levels(
        hand_lap, eye_lap, mask_gp, hand_img, eye_img,
        output_dir, reference=reference
    )
    print(f"[Debug] Level metrics keys: {list(level_metrics.keys())}")
    all_results.update(level_results)
    all_metrics.update(level_metrics)

    # Compare direct vs pyramid
    direct_results, direct_metrics = compare_direct_vs_pyramid(
        hand_img, eye_img, mask, hand_lap, eye_lap, mask_gp,
        output_dir, reference=reference
    )
    all_results.update(direct_results)
    all_metrics.update(direct_metrics)

    # Compare color spaces
    color_results, color_metrics = compare_color_spaces(
        hand_lap, eye_lap, mask_gp, hand_img, eye_img,
        output_dir, reference=reference
    )
    all_results.update(color_results)
    all_metrics.update(color_metrics)

    return all_results, all_metrics
