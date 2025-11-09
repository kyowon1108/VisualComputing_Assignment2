"""
Ablation Study: Mask Blur Kernel 변화 실험

서로 다른 blur kernel 값으로 마스크를 생성하여
블렌딩 품질이 어떻게 변하는지 분석
"""
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.preprocessing import load_and_preprocess, create_mask
from src.pyramid_generation import gaussian_pyramid_opencv, laplacian_pyramid
from src.reconstruction import blend_pyramids_at_level, reconstruct_from_laplacian
from src.blending import direct_blending
from src.metrics import calculate_metrics
from src.utils import save_image


def run_ablation_blur_kernel(blur_values=[11, 31, 51]):
    """
    Blur kernel 값 변화에 따른 ablation study

    Args:
        blur_values: 테스트할 blur kernel 값 리스트
    """
    print("\n" + "="*80)
    print(" "*20 + "ABLATION STUDY: MASK BLUR KERNEL")
    print("="*80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output', 'ablation_study')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load images
    print("\n[Step 1] Loading images...")
    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')
    hand_img, eye_img = load_and_preprocess(hand_path, eye_path)
    print(f"  ✓ Hand: {hand_img.shape}")
    print(f"  ✓ Eye: {eye_img.shape}")

    # Results storage
    all_results = {}

    # 2. Run experiments for each blur value
    for blur_kernel in blur_values:
        print(f"\n{'='*80}")
        print(f" EXPERIMENT: Blur Kernel = {blur_kernel}")
        print(f"{'='*80}")

        # Create mask with this blur kernel
        print(f"\n[Step 2] Creating mask (blur={blur_kernel})...")
        mask = create_mask(
            shape=(480, 640),
            center=(325, 315),
            axes=(48, 36),
            blur_kernel=blur_kernel,
            output_dir=None
        )
        print(f"  ✓ Mask created with blur_kernel={blur_kernel}")

        # Save mask for comparison
        mask_path = os.path.join(output_dir, f'mask_blur{blur_kernel}.jpg')
        save_image(mask, mask_path)

        # 3. Build pyramids
        print(f"\n[Step 3] Building pyramids...")
        levels = 6
        hand_gp, _ = gaussian_pyramid_opencv(hand_img, levels)
        eye_gp, _ = gaussian_pyramid_opencv(eye_img, levels)
        mask_gp, _ = gaussian_pyramid_opencv(mask, levels)

        hand_lap = laplacian_pyramid(hand_gp)
        eye_lap = laplacian_pyramid(eye_gp)

        print(f"  ✓ Pyramids built: {levels} levels")

        # 4. Direct blending (reference)
        print(f"\n[Step 4] Direct blending...")
        direct_result = direct_blending(hand_img, eye_img, mask)
        direct_path = os.path.join(output_dir, f'direct_blur{blur_kernel}.jpg')
        save_image(direct_result, direct_path)

        # 5. Pyramid blending (5-level)
        print(f"\n[Step 5] Pyramid blending (5-level)...")
        blended_lap = blend_pyramids_at_level(hand_lap, eye_lap, mask_gp, levels=None)
        pyramid_result = reconstruct_from_laplacian(
            blended_lap,
            target_shape=(480, 640),
            min_reconstruction_level=0  # Full reconstruction
        )
        pyramid_path = os.path.join(output_dir, f'pyramid_blur{blur_kernel}.jpg')
        save_image(pyramid_result, pyramid_path)

        # 6. Calculate metrics (using direct as reference)
        print(f"\n[Step 6] Calculating metrics...")
        metrics = calculate_metrics(pyramid_result, direct_result)

        # Calculate boundary smoothness
        from skimage.filters import sobel

        # Ensure mask is 2D
        mask_2d = mask if len(mask.shape) == 2 else mask[:, :, 0]

        # Extract boundary region (mask 0.2-0.8)
        boundary_mask = (mask_2d > 0.2) & (mask_2d < 0.8)

        # Calculate gradient in boundary region
        if len(pyramid_result.shape) == 3:
            pyramid_gray = np.mean(pyramid_result, axis=2)
            direct_gray = np.mean(direct_result, axis=2)
        else:
            pyramid_gray = pyramid_result
            direct_gray = direct_result

        pyramid_grad = sobel(pyramid_gray)
        direct_grad = sobel(direct_gray)

        pyramid_boundary_grad = pyramid_grad[boundary_mask]
        direct_boundary_grad = direct_grad[boundary_mask]

        boundary_metrics = {
            'pyramid_gradient_mean': float(np.mean(pyramid_boundary_grad)),
            'pyramid_gradient_std': float(np.std(pyramid_boundary_grad)),
            'direct_gradient_mean': float(np.mean(direct_boundary_grad)),
            'direct_gradient_std': float(np.std(direct_boundary_grad)),
            'boundary_pixels': int(np.sum(boundary_mask))
        }

        # Store results
        all_results[f'blur_{blur_kernel}'] = {
            'blur_kernel': blur_kernel,
            'quality_metrics': metrics,
            'boundary_metrics': boundary_metrics,
            'mask_stats': {
                'min': float(mask.min()),
                'max': float(mask.max()),
                'mean': float(mask.mean()),
                'std': float(mask.std())
            }
        }

        print(f"\n  Results for blur_kernel={blur_kernel}:")
        print(f"    SSIM: {metrics['ssim']:.4f}")
        print(f"    MSE:  {metrics['mse']:.6f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
        print(f"    Boundary Gradient Std (Pyramid): {boundary_metrics['pyramid_gradient_std']:.4f}")
        print(f"    Boundary Gradient Std (Direct):  {boundary_metrics['direct_gradient_std']:.4f}")

    # 7. Save results to JSON
    print(f"\n{'='*80}")
    print(" "*25 + "SAVING RESULTS")
    print(f"{'='*80}")

    results_path = os.path.join(output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # 8. Print comparison table
    print(f"\n{'='*80}")
    print(" "*20 + "COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"\n{'Blur':<8} {'SSIM':<10} {'MSE':<12} {'PSNR (dB)':<12} {'Boundary Grad Std':<20}")
    print("-" * 80)
    for key in sorted(all_results.keys()):
        result = all_results[key]
        blur = result['blur_kernel']
        ssim = result['quality_metrics']['ssim']
        mse = result['quality_metrics']['mse']
        psnr = result['quality_metrics']['psnr']
        grad_std = result['boundary_metrics']['pyramid_gradient_std']
        print(f"{blur:<8} {ssim:<10.4f} {mse:<12.6f} {psnr:<12.2f} {grad_std:<20.4f}")

    print(f"\n{'='*80}")
    print(" "*20 + "✓ ABLATION STUDY COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   - mask_blur*.jpg        : Masks with different blur kernels")
    print(f"   - direct_blur*.jpg      : Direct blending results")
    print(f"   - pyramid_blur*.jpg     : Pyramid blending results")
    print(f"   - ablation_results.json : Quantitative metrics (JSON)")
    print()

    return all_results


if __name__ == '__main__':
    run_ablation_blur_kernel(blur_values=[11, 31, 51])
