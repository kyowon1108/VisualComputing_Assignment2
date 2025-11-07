"""
Main pipeline for Image Pyramid Blending
"""
import os
import sys
import time
from datetime import datetime

# Import modules
from .utils import (create_output_directories, create_log_file, log_message,
                   calculate_memory_usage)
from .preprocessing import load_and_preprocess, create_mask
from .pyramid_generation import (gaussian_pyramid_opencv, gaussian_pyramid_raw,
                                laplacian_pyramid, print_pyramid_info,
                                validate_pyramid_sizes)
from .blending import direct_blending, pyramid_blending, lab_blending
from .comparison import run_all_comparisons
from .metrics import (calculate_all_metrics, save_metrics_report,
                     print_metrics_table, generate_analysis_summary)
from .visualization import (visualize_pyramid_levels, visualize_blending_comparison,
                           plot_quality_metrics, plot_histogram_comparison,
                           visualize_level_comparison)


def main():
    """
    Main pipeline for image pyramid blending

    Steps:
    1. Image loading and preprocessing
    2. Mask generation
    3. Gaussian Pyramid generation (OpenCV)
    4. Gaussian Pyramid generation (Raw convolution)
    5. Laplacian Pyramid generation
    6. Direct Blending
    7. Pyramid Blending (3/5/6 level)
    8. LAB color space Blending
    9. Performance metrics calculation
    10. Visualization and result saving
    """
    print("="*80)
    print("Image Pyramid Blending Pipeline")
    print("="*80)

    # Create output directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = create_output_directories()
    log_file = create_log_file(output_dir)

    # Input paths
    input_dir = os.path.join(base_dir, 'input')
    hand_path = os.path.join(input_dir, 'hand_raw.jpg')
    eye_path = os.path.join(input_dir, 'eye_raw.jpg')

    # Check if input files exist
    if not os.path.exists(hand_path):
        print(f"\n⚠ Warning: Hand image not found at {hand_path}")
        print("Please place your hand image at: pyramid_blending/input/hand_raw.jpg")
        return

    if not os.path.exists(eye_path):
        print(f"\n⚠ Warning: Eye image not found at {eye_path}")
        print("Please place your eye image at: pyramid_blending/input/eye_raw.jpg")
        return

    # ========================================================================
    # Phase 1: Image Loading and Preprocessing
    # ========================================================================
    print("\n[Phase 1] Image Loading and Preprocessing")
    log_message("[Phase 1] Image Loading and Preprocessing", log_file=log_file)

    hand_img, eye_img = load_and_preprocess(hand_path, eye_path, output_dir)

    print(f"  ✓ Hand image loaded: {hand_img.shape[1]}×{hand_img.shape[0]}")
    print(f"  ✓ Eye image cropped: 120×90 and placed on canvas")
    log_message(f"Hand image: {hand_img.shape}", log_file=log_file)
    log_message(f"Eye image: {eye_img.shape}", log_file=log_file)

    # Create mask
    mask = create_mask(shape=(480, 640), center=(325, 315), axes=(48, 36),
                      blur_kernel=31, output_dir=output_dir)
    print(f"  ✓ Mask created: Ellipse + Gaussian blur (kernel=71)")
    log_message("Mask created with Gaussian blur", log_file=log_file)

    # ========================================================================
    # Phase 2: Gaussian Pyramid Generation (OpenCV)
    # ========================================================================
    print("\n[Phase 2] Gaussian Pyramid Generation (OpenCV)")
    log_message("[Phase 2] Gaussian Pyramid Generation (OpenCV)", log_file=log_file)

    levels = 6

    hand_gp_cv, hand_times_cv = gaussian_pyramid_opencv(
        hand_img, levels, output_dir, 'hand')
    eye_gp_cv, eye_times_cv = gaussian_pyramid_opencv(
        eye_img, levels, output_dir, 'eye')
    mask_gp_cv, mask_times_cv = gaussian_pyramid_opencv(
        mask, levels, output_dir, 'mask')

    print_pyramid_info(hand_gp_cv, hand_times_cv, "Hand Gaussian Pyramid (OpenCV)")

    # Validate sizes
    validate_pyramid_sizes(hand_gp_cv, hand_img.shape)

    # ========================================================================
    # Phase 3: Gaussian Pyramid Generation (Raw Convolution)
    # ========================================================================
    print("\n[Phase 3] Gaussian Pyramid Generation (Raw Convolution)")
    log_message("[Phase 3] Gaussian Pyramid Generation (Raw)", log_file=log_file)

    print("  Using Gaussian kernel: [[1,4,6,4,1], ...]")

    hand_gp_raw, hand_times_raw = gaussian_pyramid_raw(
        hand_img, levels, None, 'hand_raw')  # Don't save to avoid duplication
    eye_gp_raw, eye_times_raw = gaussian_pyramid_raw(
        eye_img, levels, None, 'eye_raw')

    print_pyramid_info(hand_gp_raw, hand_times_raw, "Hand Gaussian Pyramid (Raw)")

    # ========================================================================
    # Phase 4: Laplacian Pyramid Generation
    # ========================================================================
    print("\n[Phase 4] Laplacian Pyramid Generation")
    log_message("[Phase 4] Laplacian Pyramid Generation", log_file=log_file)

    hand_lap = laplacian_pyramid(hand_gp_cv, output_dir, 'hand')
    eye_lap = laplacian_pyramid(eye_gp_cv, output_dir, 'eye')

    print(f"  ✓ Hand Laplacian: {len(hand_lap)} levels")
    print(f"  ✓ Eye Laplacian: {len(eye_lap)} levels")

    # ========================================================================
    # Phase 5: Blending
    # ========================================================================
    print("\n[Phase 5] Blending")
    log_message("[Phase 5] Blending", log_file=log_file)

    # Run all comparisons
    all_results, all_metrics = run_all_comparisons(
        hand_img, eye_img, mask, hand_lap, eye_lap, mask_gp_cv, output_dir)

    # Print metrics
    print_metrics_table(all_metrics)

    # ========================================================================
    # Phase 6: Visualization & Reports
    # ========================================================================
    print("\n[Phase 6] Visualization & Reports")
    log_message("[Phase 6] Visualization & Reports", log_file=log_file)

    viz_dir = os.path.join(output_dir, 'visualization')

    # Visualize pyramid levels
    pyramid_dict = {
        'hand_gaussian': hand_gp_cv,
        'eye_gaussian': eye_gp_cv,
        'mask_gaussian': mask_gp_cv
    }
    visualize_pyramid_levels(pyramid_dict,
                            os.path.join(viz_dir, 'pyramid_comparison.png'))

    # Visualize blending comparison
    blending_results = {
        'Direct Blending': all_results.get('direct'),
        'Pyramid (3-level)': all_results.get(3),
        'Pyramid (5-level)': all_results.get(5),
        'Pyramid (6-level)': all_results.get(6),
        'LAB Blend (5-level)': all_results.get('lab')
    }
    # Remove None values
    blending_results = {k: v for k, v in blending_results.items() if v is not None}

    visualize_blending_comparison(blending_results, all_metrics,
                                 os.path.join(viz_dir, 'blending_comparison.png'))

    # Visualize level comparison
    level_results = {k: v for k, v in all_results.items() if isinstance(k, int)}
    if level_results:
        visualize_level_comparison(level_results, all_metrics,
                                  os.path.join(viz_dir, 'level_comparison.png'))

    # Plot quality metrics
    plot_quality_metrics(all_metrics,
                        os.path.join(viz_dir, 'quality_metrics.png'))

    # Plot histograms
    hist_results = {
        'Hand Original': hand_img,
        'Direct Blend': all_results.get('direct'),
        'Pyramid Blend (5)': all_results.get(5)
    }
    hist_results = {k: v for k, v in hist_results.items() if v is not None}
    plot_histogram_comparison(hist_results,
                            os.path.join(viz_dir, 'histogram_comparison.png'))

    # ========================================================================
    # Phase 7: Save Reports
    # ========================================================================
    reports_dir = os.path.join(output_dir, 'reports')

    # Save metrics JSON
    save_metrics_report(all_metrics,
                       os.path.join(reports_dir, 'metrics.json'))

    # Generate analysis summary
    generate_analysis_summary(all_metrics,
                             os.path.join(reports_dir, 'analysis_summary.txt'))

    # Final log
    log_message("\n" + "="*80, log_file=log_file)
    log_message("Pipeline completed successfully!", log_file=log_file)
    log_message(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               log_file=log_file)

    print("\n" + "="*80)
    print("✅ All completed successfully!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Preprocessed images: {os.path.join(output_dir, 'preprocessed')}")
    print(f"  - Pyramid images: {os.path.join(output_dir, 'pyramids')}")
    print(f"  - Blending results: {os.path.join(output_dir, 'blending_results')}")
    print(f"  - Visualizations: {os.path.join(viz_dir)}")
    print(f"  - Reports: {reports_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
