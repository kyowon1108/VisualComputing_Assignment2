"""
Image blending algorithms
"""
import cv2
import numpy as np
from .reconstruction import blend_pyramids_at_level, reconstruct_from_laplacian


def direct_blending(hand, eye, mask):
    """
    Direct (Alpha) blending

    Args:
        hand: Background image (480, 640, 3)
        eye: Foreground image (480, 640, 3)
        mask: Blending mask (480, 640, 1) or (480, 640)

    Returns:
        blended: Blended image (480, 640, 3)
    """
    # Ensure mask has correct shape
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]

    # Ensure all images have same size
    if hand.shape[:2] != eye.shape[:2]:
        eye = cv2.resize(eye, (hand.shape[1], hand.shape[0]))

    if mask.shape[:2] != hand.shape[:2]:
        mask = cv2.resize(mask, (hand.shape[1], hand.shape[0]))
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

    # Direct blending: output = hand * (1 - mask) + eye * mask
    blended = hand * (1 - mask) + eye * mask

    return blended


def pyramid_blending(hand_lap, eye_lap, mask_gp, levels=5):
    """
    Pyramid blending using Laplacian pyramids

    Args:
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        levels: Number of pyramid levels to use

    Returns:
        blended: Blended image
    """
    # Blend pyramids
    blended_lap = blend_pyramids_at_level(hand_lap, eye_lap, mask_gp, levels)

    # Reconstruct from blended Laplacian pyramid
    blended = reconstruct_from_laplacian(blended_lap)

    return blended


def lab_blending(hand_lap, eye_lap, mask_gp, hand_rgb, eye_rgb, levels=5):
    """
    LAB color space blending

    Args:
        hand_lap: Hand Laplacian pyramid (RGB)
        eye_lap: Eye Laplacian pyramid (RGB)
        mask_gp: Mask Gaussian pyramid
        hand_rgb: Original hand image (for color reference)
        eye_rgb: Original eye image (for color reference)
        levels: Number of pyramid levels to use

    Returns:
        blended: Blended image in RGB
    """
    # Convert RGB to LAB
    hand_lab = rgb_to_lab(hand_rgb)
    eye_lab = rgb_to_lab(eye_rgb)

    # Split LAB channels
    hand_l, hand_a, hand_b = cv2.split(hand_lab)
    eye_l, eye_a, eye_b = cv2.split(eye_lab)

    # Build Laplacian pyramids for L channel only
    from .pyramid_generation import build_pyramids

    hand_l_gp, hand_l_lap, _ = build_pyramids(hand_l, levels, method='opencv')
    eye_l_gp, eye_l_lap, _ = build_pyramids(eye_l, levels, method='opencv')

    # Blend L channel using pyramid blending
    blended_l_lap = blend_pyramids_at_level(hand_l_lap, eye_l_lap, mask_gp, levels)
    blended_l = reconstruct_from_laplacian(blended_l_lap)

    # For a and b channels, use direct blending
    mask_2d = mask_gp[0]
    if len(mask_2d.shape) == 3:
        mask_2d = mask_2d[:, :, 0]

    blended_a = hand_a * (1 - mask_2d) + eye_a * mask_2d
    blended_b = hand_b * (1 - mask_2d) + eye_b * mask_2d

    # Merge LAB channels
    blended_lab = cv2.merge([blended_l, blended_a, blended_b])

    # Convert back to RGB
    blended_rgb = lab_to_rgb(blended_lab)

    return blended_rgb


def rgb_to_lab(rgb_image):
    """
    Convert RGB image to LAB color space

    Args:
        rgb_image: RGB image (float32, [0, 1]) or (uint8, [0, 255])

    Returns:
        lab_image: LAB image
    """
    # Ensure float32 in [0, 1]
    if rgb_image.dtype == np.uint8:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    # Clip to valid range
    rgb_image = np.clip(rgb_image, 0, 1)

    # Convert to uint8 for OpenCV
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # Convert BGR to LAB
    lab_image = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB)

    # Normalize LAB to [0, 1]
    # L: 0-100 -> 0-1
    # a, b: 0-255 (with 128 as neutral) -> 0-1
    lab_image = lab_image.astype(np.float32)
    lab_image[:, :, 0] = lab_image[:, :, 0] / 100.0  # L channel
    lab_image[:, :, 1] = lab_image[:, :, 1] / 255.0  # a channel
    lab_image[:, :, 2] = lab_image[:, :, 2] / 255.0  # b channel

    return lab_image


def lab_to_rgb(lab_image):
    """
    Convert LAB image to RGB color space

    Args:
        lab_image: LAB image (normalized [0, 1])

    Returns:
        rgb_image: RGB image (float32, [0, 1])
    """
    # Denormalize LAB
    lab_denorm = lab_image.copy()
    lab_denorm[:, :, 0] = np.clip(lab_denorm[:, :, 0] * 100.0, 0, 100)  # L
    lab_denorm[:, :, 1] = np.clip(lab_denorm[:, :, 1] * 255.0, 0, 255)  # a
    lab_denorm[:, :, 2] = np.clip(lab_denorm[:, :, 2] * 255.0, 0, 255)  # b

    # Convert to uint8
    lab_uint8 = lab_denorm.astype(np.uint8)

    # Convert LAB to BGR
    bgr_uint8 = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)

    # Convert BGR to RGB
    rgb_uint8 = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    rgb_image = rgb_uint8.astype(np.float32) / 255.0

    return rgb_image


def yuv_blending(hand_lap, eye_lap, mask_gp, hand_rgb, eye_rgb, levels=5):
    """
    YUV color space blending

    Args:
        hand_lap: Hand Laplacian pyramid (RGB)
        eye_lap: Eye Laplacian pyramid (RGB)
        mask_gp: Mask Gaussian pyramid
        hand_rgb: Original hand image (for color reference)
        eye_rgb: Original eye image (for color reference)
        levels: Number of pyramid levels to use

    Returns:
        blended: Blended image in RGB
    """
    # Convert RGB to YUV
    hand_yuv = rgb_to_yuv(hand_rgb)
    eye_yuv = rgb_to_yuv(eye_rgb)

    # Split YUV channels
    hand_y, hand_u, hand_v = cv2.split(hand_yuv)
    eye_y, eye_u, eye_v = cv2.split(eye_yuv)

    # Build Laplacian pyramids for Y channel only
    from .pyramid_generation import build_pyramids

    hand_y_gp, hand_y_lap, _ = build_pyramids(hand_y, levels, method='opencv')
    eye_y_gp, eye_y_lap, _ = build_pyramids(eye_y, levels, method='opencv')

    # Blend Y channel using pyramid blending
    blended_y_lap = blend_pyramids_at_level(hand_y_lap, eye_y_lap, mask_gp, levels)
    blended_y = reconstruct_from_laplacian(blended_y_lap)

    # For U and V channels, use direct blending
    mask_2d = mask_gp[0]
    if len(mask_2d.shape) == 3:
        mask_2d = mask_2d[:, :, 0]

    blended_u = hand_u * (1 - mask_2d) + eye_u * mask_2d
    blended_v = hand_v * (1 - mask_2d) + eye_v * mask_2d

    # Merge YUV channels
    blended_yuv = cv2.merge([blended_y, blended_u, blended_v])

    # Convert back to RGB
    blended_rgb = yuv_to_rgb(blended_yuv)

    return blended_rgb


def rgb_to_yuv(rgb_image):
    """
    Convert RGB image to YUV color space

    Args:
        rgb_image: RGB image (float32, [0, 1]) or (uint8, [0, 255])

    Returns:
        yuv_image: YUV image (normalized [0, 1])
    """
    # Ensure float32 in [0, 1]
    if rgb_image.dtype == np.uint8:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    # Clip to valid range
    rgb_image = np.clip(rgb_image, 0, 1)

    # Convert to uint8 for OpenCV
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # Convert BGR to YUV
    yuv_image = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2YUV)

    # Normalize to [0, 1]
    yuv_image = yuv_image.astype(np.float32) / 255.0

    return yuv_image


def yuv_to_rgb(yuv_image):
    """
    Convert YUV image to RGB color space

    Args:
        yuv_image: YUV image (normalized [0, 1])

    Returns:
        rgb_image: RGB image (float32, [0, 1])
    """
    # Denormalize to [0, 255]
    yuv_uint8 = (np.clip(yuv_image, 0, 1) * 255).astype(np.uint8)

    # Convert YUV to BGR
    bgr_uint8 = cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2BGR)

    # Convert BGR to RGB
    rgb_uint8 = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    rgb_image = rgb_uint8.astype(np.float32) / 255.0

    return rgb_image


def multi_level_blending(hand_lap, eye_lap, mask_gp, level_configs):
    """
    Perform blending with different level configurations

    Args:
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        level_configs: List of level counts to try [3, 5, 6]

    Returns:
        results: Dictionary mapping level count to blended image
    """
    results = {}

    for levels in level_configs:
        blended = pyramid_blending(hand_lap, eye_lap, mask_gp, levels)
        results[levels] = blended

    return results
