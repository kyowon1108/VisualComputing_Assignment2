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

    IMPORTANT: Each level comparison must use pyramids built with that specific
    number of levels. Using a 6-level pyramid for 3-level blending will fail
    because the base level won't be the proper Gaussian base.

    Args:
        hand_lap: Hand Laplacian pyramid (ignored, will rebuild for each level)
        eye_lap: Eye Laplacian pyramid (ignored, will rebuild for each level)
        mask_gp: Mask Gaussian pyramid (ignored, will rebuild for each level)
        hand_rgb: Hand RGB image (for reference)
        eye_rgb: Eye RGB image (for reference)
        output_dir: Output directory
        reference: Optional reference image for metrics

    Returns:
        results: Dictionary mapping level count to blended image
        metrics_dict: Dictionary of metrics
    """
    from .pyramid_generation import gaussian_pyramid_opencv, laplacian_pyramid

    print("\n[Comparison] Pyramid Reconstruction Levels (0-5)")

    results = {}
    metrics_dict = {}

    # Build ONE 6-level pyramid for everything
    max_levels = 6

    # Create mask from hand_rgb
    import cv2
    import numpy as np
    from .preprocessing import create_mask

    mask = create_mask(shape=(hand_rgb.shape[0], hand_rgb.shape[1]),
                      center=(325, 315), axes=(48, 36),
                      blur_kernel=31, output_dir=None)

    print(f"  Building {max_levels}-level pyramid...")
    hand_gp, _ = gaussian_pyramid_opencv(hand_rgb, max_levels)
    eye_gp, _ = gaussian_pyramid_opencv(eye_rgb, max_levels)
    mask_gp_full, _ = gaussian_pyramid_opencv(mask, max_levels)

    hand_lap_full = laplacian_pyramid(hand_gp)
    eye_lap_full = laplacian_pyramid(eye_gp)

    # Blend the pyramids (once)
    from .reconstruction import blend_pyramids_at_level, reconstruct_from_laplacian
    blended_lap = blend_pyramids_at_level(hand_lap_full, eye_lap_full,
                                         mask_gp_full, levels=None)

    # Save blended Laplacian pyramid levels for visualization
    print(f"  Saving blended Laplacian pyramid levels...")

    # Create blend_laplacian directory
    blend_lap_dir = os.path.join(output_dir, 'pyramids', 'blend_laplacian')
    os.makedirs(blend_lap_dir, exist_ok=True)

    for level in range(len(blended_lap)):
        lap_level = blended_lap[level]

        # Normalize Laplacian for visualization
        # Laplacian values are centered around 0, need to shift to [0, 1]
        lap_normalized = lap_level.copy()

        # Shift and scale to make visible
        # Most values are near 0, so we enhance contrast
        lap_min = lap_normalized.min()
        lap_max = lap_normalized.max()
        lap_range = lap_max - lap_min

        if lap_range > 0:
            # Normalize to [0, 1]
            lap_normalized = (lap_normalized - lap_min) / lap_range
        else:
            lap_normalized = np.zeros_like(lap_normalized)

        # Resize to 640×480 for consistent visualization
        if lap_normalized.shape[:2] != (480, 640):
            lap_display = cv2.resize(lap_normalized, (640, 480))
        else:
            lap_display = lap_normalized

        # Save to pyramids/blend_laplacian/
        lap_path = os.path.join(blend_lap_dir, f'laplacian_level_{level}.jpg')
        save_image(lap_display, lap_path)

    print(f"    ✓ Saved {len(blended_lap)} Laplacian levels")

    # Now reconstruct to different stopping levels
    for stop_level in range(6):
        print(f"  Generating pyramid_blend_{stop_level}level.jpg (stop reconstruction at level {stop_level})...")

        # Reconstruct with stopping point
        blended = reconstruct_from_laplacian(blended_lap,
                                            target_shape=(480, 640),
                                            stop_at_level=stop_level)

        results[stop_level] = blended

        # Save result with new naming: pyramid_blend_0level.jpg ~ pyramid_blend_5level.jpg
        output_path = os.path.join(output_dir, 'blending_results',
                                  f'pyramid_blend_{stop_level}level.jpg')
        save_image(blended, output_path)

        # Calculate metrics if reference provided
        if reference is not None:
            metrics = calculate_metrics(blended, reference)
            metrics_dict[f'{stop_level}level'] = metrics
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
    # NOTE: This is now handled by compare_pyramid_levels
    # Don't save here to avoid overwriting the correct result
    print("  (5-level pyramid already tested in compare_pyramid_levels)")

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
