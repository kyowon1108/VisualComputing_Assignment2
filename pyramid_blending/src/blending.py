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
