"""
Generate gray-background Laplacian pyramid visualizations
Makes Laplacian pyramids easier to see by centering around gray instead of black

This version uses the proper preprocessing pipeline:
- Hand: 640x480 center-cropped and resized
- Eye: 120x90 cropped/resized and placed on 640x480 black canvas at position (325, 315)
- Mask: Elliptical mask at (325, 315) with blur
- Masked versions: hand*mask and eye*(1-mask)
"""
import cv2
import numpy as np
import os
from src.utils import save_image
from src.preprocessing import load_and_preprocess, create_mask
from src.pyramid_generation import gaussian_pyramid_opencv, laplacian_pyramid


def save_laplacian_with_gray_background(laplacian_pyramid, output_dir, name='image'):
    """
    Save Laplacian pyramid with gray background for better visibility

    For Laplacian pyramids:
    - They contain both positive and negative values (representing details)
    - 0 = no detail
    - Positive values = lighter details
    - Negative values = darker details

    Gray background visualization:
    - 0 (no detail) → 0.5 (gray)
    - Positive details → lighter than gray (0.5 to 1.0)
    - Negative details → darker than gray (0.0 to 0.5)

    Args:
        laplacian_pyramid: List of Laplacian pyramid levels
        output_dir: Output directory
        name: Name prefix for saved images
    """
    pyramid_dir = os.path.join(output_dir, f'{name}_laplacian_gray')
    os.makedirs(pyramid_dir, exist_ok=True)

    for i, lap_level in enumerate(laplacian_pyramid):
        # For the last level (which is the Gaussian base), just normalize normally
        if i == len(laplacian_pyramid) - 1:
            lap_normalized = lap_level.copy()
            # Standard min-max normalization for Gaussian base
            if lap_normalized.max() > lap_normalized.min():
                lap_normalized = (lap_normalized - lap_normalized.min()) / \
                                (lap_normalized.max() - lap_normalized.min())
        else:
            # For Laplacian levels, center around 0.5 (gray)
            lap_normalized = lap_level.copy()

            # Find the maximum absolute value for symmetric scaling
            max_abs = max(abs(lap_normalized.min()), abs(lap_normalized.max()))

            if max_abs > 1e-10:  # Avoid division by zero
                # Center around 0.5 with symmetric range
                # -max_abs → 0.0 (black)
                # 0 → 0.5 (gray)
                # +max_abs → 1.0 (white)
                lap_normalized = 0.5 + lap_normalized / (2.0 * max_abs)

                # Clamp to [0, 1] to handle any numerical issues
                lap_normalized = np.clip(lap_normalized, 0.0, 1.0)
            else:
                # If nearly zero everywhere, just use gray
                lap_normalized = np.ones_like(lap_normalized) * 0.5

        # Save the normalized image
        output_path = os.path.join(pyramid_dir, f'level_{i}.jpg')
        save_image(lap_normalized, output_path)
        print(f"  ✓ Saved {name} Laplacian level {i} with gray background")

    print(f"  Total: {len(laplacian_pyramid)} levels saved to {pyramid_dir}")


def main():
    """Generate gray-background Laplacian pyramids using proper preprocessing"""
    print("\n" + "="*60)
    print("Gray-Background Laplacian Pyramids (with Preprocessing)")
    print("="*60)

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output', 'pyramids')

    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')

    # Load and preprocess using the same pipeline as main.py
    print(f"\nPreprocessing images...")
    hand_img, eye_img = load_and_preprocess(hand_path, eye_path, output_dir=None)
    print(f"  ✓ Hand: {hand_img.shape} (640x480 center-cropped)")
    print(f"  ✓ Eye: {eye_img.shape} (120x90 on 640x480 black canvas)")

    # Create mask
    print(f"\nCreating mask...")
    mask = create_mask(shape=(480, 640),
                      center=(325, 315),
                      axes=(48, 36),
                      blur_kernel=31,
                      output_dir=None)
    print(f"  ✓ Mask: {mask.shape} (ellipse at (325, 315) with blur)")

    # Generate pyramids
    levels = 6
    print(f"\nBuilding {levels}-level pyramids...")

    # 1. Hand pyramids
    print("\n1. Hand Laplacian Pyramid:")
    hand_gp, _ = gaussian_pyramid_opencv(hand_img, levels)
    hand_lap = laplacian_pyramid(hand_gp)
    save_laplacian_with_gray_background(hand_lap, output_dir, 'hand')

    # 2. Eye pyramids
    print("\n2. Eye Laplacian Pyramid:")
    eye_gp, _ = gaussian_pyramid_opencv(eye_img, levels)
    eye_lap = laplacian_pyramid(eye_gp)
    save_laplacian_with_gray_background(eye_lap, output_dir, 'eye')

    # 3. Mask pyramid
    print("\n3. Mask Gaussian Pyramid:")
    mask_gp, _ = gaussian_pyramid_opencv(mask, levels)
    print(f"  ✓ Generated {len(mask_gp)} mask levels")

    # 4. Masked hand (hand * mask)
    print("\n4. Masked Hand Laplacian (hand × mask):")
    hand_masked = hand_img * mask
    hand_masked_gp, _ = gaussian_pyramid_opencv(hand_masked, levels)
    hand_masked_lap = laplacian_pyramid(hand_masked_gp)
    save_laplacian_with_gray_background(hand_masked_lap, output_dir, 'hand_masked')

    # 5. Masked eye (eye * (1 - mask))
    print("\n5. Masked Eye Laplacian (eye × (1-mask)):")
    eye_masked = eye_img * (1.0 - mask)
    eye_masked_gp, _ = gaussian_pyramid_opencv(eye_masked, levels)
    eye_masked_lap = laplacian_pyramid(eye_masked_gp)
    save_laplacian_with_gray_background(eye_masked_lap, output_dir, 'eye_masked')

    # 6. Blended pyramids
    print("\n6. Blended Laplacian Pyramid:")
    from src.reconstruction import blend_pyramids_at_level
    blended_lap = blend_pyramids_at_level(hand_lap, eye_lap, mask_gp, levels=None)
    save_laplacian_with_gray_background(blended_lap, output_dir, 'blend')

    print("\n" + "="*60)
    print("✓ All gray-background Laplacian pyramids generated!")
    print("="*60)
    print(f"\nOutput directories:")
    print(f"  - {os.path.join(output_dir, 'hand_laplacian_gray')}")
    print(f"  - {os.path.join(output_dir, 'eye_laplacian_gray')}")
    print(f"  - {os.path.join(output_dir, 'hand_masked_laplacian_gray')}")
    print(f"  - {os.path.join(output_dir, 'eye_masked_laplacian_gray')}")
    print(f"  - {os.path.join(output_dir, 'blend_laplacian_gray')}")


if __name__ == '__main__':
    main()
