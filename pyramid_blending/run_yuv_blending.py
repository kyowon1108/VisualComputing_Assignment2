"""
YUV Color Space Blending 실행 스크립트
"""
import os
from src.preprocessing import load_and_preprocess, create_mask
from src.pyramid_generation import gaussian_pyramid_opencv, laplacian_pyramid
from src.blending import yuv_blending
from src.utils import save_image
from src.metrics import calculate_metrics


def main():
    """Generate YUV blending result"""
    print("\n" + "="*80)
    print(" "*20 + "YUV COLOR SPACE BLENDING")
    print("="*80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output', 'blending_results')

    # 1. Load images
    print("\n[Step 1] Loading images...")
    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')
    hand_img, eye_img = load_and_preprocess(hand_path, eye_path)
    print(f"  ✓ Hand: {hand_img.shape}")
    print(f"  ✓ Eye: {eye_img.shape}")

    # 2. Create mask
    print("\n[Step 2] Creating mask...")
    mask = create_mask(shape=(480, 640),
                      center=(325, 315),
                      axes=(48, 36),
                      blur_kernel=31,
                      output_dir=None)
    print(f"  ✓ Mask: {mask.shape}")

    # 3. Build pyramids
    print("\n[Step 3] Building pyramids...")
    levels = 6
    hand_gp, _ = gaussian_pyramid_opencv(hand_img, levels)
    eye_gp, _ = gaussian_pyramid_opencv(eye_img, levels)
    mask_gp, _ = gaussian_pyramid_opencv(mask, levels)

    hand_lap = laplacian_pyramid(hand_gp)
    eye_lap = laplacian_pyramid(eye_gp)
    print(f"  ✓ Pyramids built: {levels} levels")

    # 4. YUV blending
    print("\n[Step 4] YUV blending (5-level)...")
    yuv_result = yuv_blending(hand_lap, eye_lap, mask_gp, hand_img, eye_img, 5)

    # Save result
    yuv_path = os.path.join(output_dir, 'yuv_blend_5level.jpg')
    save_image(yuv_result, yuv_path)
    print(f"  ✓ Saved: {yuv_path}")

    # 5. Calculate metrics (compare with LAB)
    lab_path = os.path.join(output_dir, 'lab_blend_5level.jpg')
    if os.path.exists(lab_path):
        from src.utils import load_image
        lab_result = load_image(lab_path)

        metrics = calculate_metrics(yuv_result, lab_result)
        print(f"\n  Comparison with LAB:")
        print(f"    SSIM: {metrics['ssim']:.4f}")
        print(f"    MSE:  {metrics['mse']:.6f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")

    print("\n" + "="*80)
    print(" "*20 + "✓ YUV BLENDING COMPLETE!")
    print("="*80)
    print(f"\n📁 Result saved to: {yuv_path}")
    print()


if __name__ == '__main__':
    main()
