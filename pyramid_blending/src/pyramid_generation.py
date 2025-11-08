"""
Pyramid generation module: Gaussian and Laplacian pyramids
"""
import cv2
import numpy as np
import time
import os
from .utils import save_image, gaussian_kernel_5x5, convolve2d


def gaussian_pyramid_opencv(image, levels=5, output_dir=None, name='image'):
    """
    Generate Gaussian pyramid using OpenCV

    Args:
        image: Input image (H, W, 3) or (H, W)
        levels: Number of pyramid levels
        output_dir: Optional directory to save pyramid images
        name: Name prefix for saved images

    Returns:
        gp: List of images [level_0, level_1, ..., level_n]
        times: List of processing times for each level
    """
    gp = []
    times = []

    current = image.copy()
    gp.append(current)
    times.append(0)  # Level 0 is just the original

    for i in range(1, levels):
        start_time = time.time()
        current = cv2.pyrDown(current)
        elapsed = (time.time() - start_time) * 1000  # milliseconds

        gp.append(current)
        times.append(elapsed)

        # Save if output_dir provided
        if output_dir:
            pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_gaussian')
            save_image(current, os.path.join(pyramid_dir, f'level_{i}.png'))

    # Save level 0
    if output_dir:
        pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_gaussian')
        save_image(gp[0], os.path.join(pyramid_dir, 'level_0.png'))

    return gp, times


def gaussian_pyramid_raw(image, levels=5, output_dir=None, name='image_raw'):
    """
    Generate Gaussian pyramid using raw convolution

    Args:
        image: Input image (H, W, 3) or (H, W)
        levels: Number of pyramid levels
        output_dir: Optional directory to save pyramid images
        name: Name prefix for saved images

    Returns:
        gp: List of images [level_0, level_1, ..., level_n]
        times: List of processing times for each level
    """
    gp = []
    times = []

    # Get Gaussian kernel
    kernel = gaussian_kernel_5x5()

    current = image.copy()
    gp.append(current)
    times.append(0)  # Level 0 is just the original

    for i in range(1, levels):
        start_time = time.time()

        # Step 1: Gaussian convolution
        blurred = convolve2d(current, kernel)

        # Step 2: Subsample (stride=2)
        subsampled = blurred[::2, ::2]

        elapsed = (time.time() - start_time) * 1000  # milliseconds

        gp.append(subsampled)
        times.append(elapsed)

        current = subsampled

        # Save if output_dir provided
        if output_dir:
            pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_gaussian')
            save_image(subsampled, os.path.join(pyramid_dir, f'level_{i}.png'))

    # Save level 0
    if output_dir:
        pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_gaussian')
        save_image(gp[0], os.path.join(pyramid_dir, 'level_0.png'))

    return gp, times


def laplacian_pyramid(gaussian_pyramid, output_dir=None, name='image'):
    """
    Generate Laplacian pyramid from Gaussian pyramid using OpenCV

    Args:
        gaussian_pyramid: List of Gaussian pyramid levels
        output_dir: Optional directory to save pyramid images
        name: Name prefix for saved images

    Returns:
        lp: List of Laplacian images [L0, L1, ..., Ln-1, Gn]
    """
    lp = []
    levels = len(gaussian_pyramid)

    # For each level except the last
    for i in range(levels - 1):
        # Get current Gaussian level
        G_i = gaussian_pyramid[i]

        # Get next Gaussian level and upsample it
        G_i1 = gaussian_pyramid[i + 1]
        G_i1_upsampled = cv2.pyrUp(G_i1)

        # Ensure same size (handle rounding issues)
        if G_i1_upsampled.shape[:2] != G_i.shape[:2]:
            G_i1_upsampled = cv2.resize(G_i1_upsampled,
                                        (G_i.shape[1], G_i.shape[0]))

        # Laplacian = G_i - upsample(G_i+1)
        L_i = G_i - G_i1_upsampled
        lp.append(L_i)

        # Save if output_dir provided
        if output_dir:
            pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_laplacian')
            save_image(L_i, os.path.join(pyramid_dir, f'level_{i}.png'))

    # Add the smallest Gaussian level as the last Laplacian level
    lp.append(gaussian_pyramid[-1])

    # Save last level
    if output_dir:
        pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_laplacian')
        save_image(gaussian_pyramid[-1],
                  os.path.join(pyramid_dir, f'level_{levels-1}.png'))

    return lp


def laplacian_pyramid_raw(gaussian_pyramid, output_dir=None, name='image_raw'):
    """
    Generate Laplacian pyramid from Gaussian pyramid using raw implementation

    This is the raw implementation without OpenCV dependencies:
    - Uses custom upsample_raw() instead of cv2.pyrUp()
    - Upsampling: zero-insertion + Gaussian convolution + 4x normalization

    Algorithm:
    For each level i (except last):
        L[i] = G[i] - upsample(G[i+1])
    L[n] = G[n]  (residual: smallest Gaussian level)

    Args:
        gaussian_pyramid: List of Gaussian pyramid levels
        output_dir: Optional directory to save pyramid images
        name: Name prefix for saved images

    Returns:
        lp: List of Laplacian images [L0, L1, ..., Ln-1, Gn]
    """
    from .utils import gaussian_kernel_5x5, upsample_raw

    lp = []
    levels = len(gaussian_pyramid)
    kernel = gaussian_kernel_5x5()

    # For each level except the last
    for i in range(levels - 1):
        # Get current Gaussian level
        G_i = gaussian_pyramid[i]

        # Get next Gaussian level and upsample it using raw implementation
        G_i1 = gaussian_pyramid[i + 1]
        G_i1_upsampled = upsample_raw(G_i1, kernel)

        # Ensure same size (handle rounding issues from odd dimensions)
        if G_i1_upsampled.shape[:2] != G_i.shape[:2]:
            # Crop to match (upsampling can create slightly larger image)
            h, w = G_i.shape[:2]
            G_i1_upsampled = G_i1_upsampled[:h, :w]
            if len(G_i.shape) == 3:
                G_i1_upsampled = G_i1_upsampled[:, :, :G_i.shape[2]]

        # Laplacian = G_i - upsample(G_i+1)
        # This captures the details lost during downsampling
        L_i = G_i - G_i1_upsampled
        lp.append(L_i)

        # Save if output_dir provided
        if output_dir:
            pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_laplacian')
            save_image(L_i, os.path.join(pyramid_dir, f'level_{i}.png'))

    # Add the smallest Gaussian level as the last Laplacian level
    # This is the residual (base) that cannot be decomposed further
    lp.append(gaussian_pyramid[-1])

    # Save last level
    if output_dir:
        pyramid_dir = os.path.join(output_dir, 'pyramids', f'{name}_laplacian')
        save_image(gaussian_pyramid[-1],
                  os.path.join(pyramid_dir, f'level_{levels-1}.png'))

    return lp


def build_pyramids(image, levels=5, method='opencv', output_dir=None, name='image'):
    """
    Build both Gaussian and Laplacian pyramids

    Args:
        image: Input image
        levels: Number of pyramid levels
        method: 'opencv' or 'raw'
        output_dir: Optional directory to save pyramid images
        name: Name prefix for saved images

    Returns:
        gaussian_pyr: Gaussian pyramid
        laplacian_pyr: Laplacian pyramid
        times: Processing times
    """
    if method == 'opencv':
        gaussian_pyr, times = gaussian_pyramid_opencv(image, levels, output_dir, name)
        laplacian_pyr = laplacian_pyramid(gaussian_pyr, output_dir, name)
    elif method == 'raw':
        gaussian_pyr, times = gaussian_pyramid_raw(image, levels, output_dir, name)
        laplacian_pyr = laplacian_pyramid_raw(gaussian_pyr, output_dir, name)
    else:
        raise ValueError(f"Unknown method: {method}")

    return gaussian_pyr, laplacian_pyr, times


def print_pyramid_info(gaussian_pyr, times, method_name='Gaussian Pyramid'):
    """
    Print pyramid information

    Args:
        gaussian_pyr: Gaussian pyramid
        times: Processing times
        method_name: Name of the method
    """
    print(f"\n{method_name}:")
    total_memory = 0

    for i, (level, time_ms) in enumerate(zip(gaussian_pyr, times)):
        memory = level.nbytes / (1024 * 1024)  # MB
        total_memory += memory
        print(f"  ✓ Level {i}: {level.shape} - Time: {time_ms:.2f}ms")

    print(f"  Total Memory: {total_memory:.2f} MB")


def validate_pyramid_sizes(pyramid, expected_base_shape):
    """
    Validate pyramid level sizes

    Args:
        pyramid: Pyramid to validate
        expected_base_shape: Expected shape of level 0

    Returns:
        valid: True if all sizes are correct
    """
    # Check level 0
    if pyramid[0].shape[:2] != expected_base_shape[:2]:
        print(f"Warning: Level 0 shape mismatch: {pyramid[0].shape} vs {expected_base_shape}")
        return False

    # Check each subsequent level is approximately half the previous
    for i in range(1, len(pyramid)):
        prev_h, prev_w = pyramid[i-1].shape[:2]
        curr_h, curr_w = pyramid[i].shape[:2]

        expected_h = (prev_h + 1) // 2
        expected_w = (prev_w + 1) // 2

        if abs(curr_h - expected_h) > 1 or abs(curr_w - expected_w) > 1:
            print(f"Warning: Level {i} size incorrect: {curr_h}x{curr_w} vs expected {expected_h}x{expected_w}")
            return False

    return True
