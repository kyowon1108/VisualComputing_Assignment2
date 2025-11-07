"""
Pyramid reconstruction module
"""
import cv2
import numpy as np


def reconstruct_from_laplacian(lap_pyramid, target_shape=None):
    """
    Reconstruct image from Laplacian pyramid

    Args:
        lap_pyramid: List of Laplacian levels [L0, L1, ..., Ln-1, Gn]
        target_shape: Optional target shape for final image (H, W)

    Returns:
        reconstructed: Reconstructed image
    """
    # Start from the smallest level (last element)
    result = lap_pyramid[-1].copy()

    # Reconstruct from bottom to top
    for i in range(len(lap_pyramid) - 2, -1, -1):
        # Upsample result
        result = cv2.pyrUp(result)

        # Get target Laplacian level
        L_i = lap_pyramid[i]

        # Ensure same size (handle rounding issues)
        if result.shape[:2] != L_i.shape[:2]:
            result = cv2.resize(result, (L_i.shape[1], L_i.shape[0]))

        # Add Laplacian
        result = result + L_i

    # If target shape is specified, resize to match
    if target_shape is not None:
        if result.shape[:2] != target_shape:
            result = cv2.resize(result, (target_shape[1], target_shape[0]))

    return result


def validate_reconstruction(original, reconstructed, tolerance=0.01):
    """
    Validate reconstruction quality

    Args:
        original: Original image
        reconstructed: Reconstructed image
        tolerance: Maximum allowed relative error

    Returns:
        valid: True if reconstruction is within tolerance
        error: Reconstruction error
    """
    # Ensure same size
    if original.shape != reconstructed.shape:
        reconstructed = cv2.resize(reconstructed,
                                  (original.shape[1], original.shape[0]))

    # Calculate error
    diff = np.abs(original - reconstructed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Calculate relative error
    if original.dtype == np.float32 or original.dtype == np.float64:
        max_value = 1.0
    else:
        max_value = 255.0

    relative_error = mean_diff / max_value

    print(f"\nReconstruction Validation:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Relative error: {relative_error:.4%}")

    valid = relative_error < tolerance

    if valid:
        print(f"  ✓ Reconstruction valid (error < {tolerance:.2%})")
    else:
        print(f"  ✗ Reconstruction error exceeds tolerance ({tolerance:.2%})")

    return valid, relative_error


def blend_pyramids_at_level(lap1, lap2, mask_gp, levels=None):
    """
    Blend two Laplacian pyramids using mask Gaussian pyramid

    Args:
        lap1: First Laplacian pyramid (background)
        lap2: Second Laplacian pyramid (foreground)
        mask_gp: Mask Gaussian pyramid
        levels: Number of levels to blend (if None, use all)

    Returns:
        blended_lap: Blended Laplacian pyramid
    """
    if levels is None:
        levels = len(lap1)

    blended_lap = []

    for i in range(levels):
        # Get mask for this level
        mask = mask_gp[i]

        # Ensure mask has correct shape
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

        # Get Laplacian levels
        L1 = lap1[i]
        L2 = lap2[i]

        # Ensure same size
        if L1.shape[:2] != L2.shape[:2]:
            L2 = cv2.resize(L2, (L1.shape[1], L1.shape[0]))

        if mask.shape[:2] != L1.shape[:2]:
            mask = cv2.resize(mask, (L1.shape[1], L1.shape[0]))
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]

        # Match mask dimensions to Laplacian dimensions
        # If L1 is 2D (grayscale), mask should be 2D
        # If L1 is 3D (color), mask should be 3D
        if len(L1.shape) == 2 and len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Remove channel dimension
        elif len(L1.shape) == 3 and len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]  # Add channel dimension

        # Blend: L_blended = L1 * (1 - mask) + L2 * mask
        L_blended = L1 * (1 - mask) + L2 * mask

        blended_lap.append(L_blended)

    return blended_lap


def print_pyramid_stats(pyramid, name='Pyramid'):
    """
    Print statistics about pyramid

    Args:
        pyramid: Pyramid to analyze
        name: Name for display
    """
    print(f"\n{name} Statistics:")
    for i, level in enumerate(pyramid):
        min_val = np.min(level)
        max_val = np.max(level)
        mean_val = np.mean(level)
        print(f"  Level {i}: shape={level.shape}, "
              f"range=[{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
