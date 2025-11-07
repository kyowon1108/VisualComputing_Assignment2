"""
Image preprocessing module for pyramid blending
"""
import cv2
import numpy as np
from .utils import normalize_image, save_image


def load_and_preprocess(hand_path, eye_path, output_dir=None):
    """
    Load and preprocess hand and eye images

    Args:
        hand_path: Path to hand image
        eye_path: Path to eye image
        output_dir: Optional output directory to save preprocessed images

    Returns:
        hand_img: (480, 640, 3) dtype=float32, normalized to [0, 1]
        eye_img: (480, 640, 3) dtype=float32, eye positioned at center
    """
    # Load images
    hand_raw = cv2.imread(hand_path)
    eye_raw = cv2.imread(eye_path)

    if hand_raw is None:
        raise FileNotFoundError(f"Hand image not found: {hand_path}")
    if eye_raw is None:
        raise FileNotFoundError(f"Eye image not found: {eye_path}")

    # Convert BGR to RGB
    hand_raw = cv2.cvtColor(hand_raw, cv2.COLOR_BGR2RGB)
    eye_raw = cv2.cvtColor(eye_raw, cv2.COLOR_BGR2RGB)

    # 1. Process hand image: center crop and resize to 640×480
    hand_img = preprocess_hand(hand_raw, target_size=(640, 480))

    # 2. Process eye image: crop and resize to 120×90
    eye_cropped = preprocess_eye(eye_raw, target_size=(120, 90))

    # 3. Place eye on 640×480 canvas at position (325, 315)
    # Position (325, 315) means row=325, col=315
    eye_img = place_on_canvas(eye_cropped, canvas_size=(480, 640),
                              position=(325, 315))  # Eye position

    # Normalize to [0, 1]
    hand_img = normalize_image(hand_img)
    eye_img = normalize_image(eye_img)

    # Save preprocessed images if output_dir is provided
    if output_dir:
        import os
        preprocessed_dir = os.path.join(output_dir, 'preprocessed')
        save_image(hand_img, os.path.join(preprocessed_dir, 'hand_640x480.jpg'))
        save_image(eye_img, os.path.join(preprocessed_dir, 'eye_120x90.jpg'))

    return hand_img, eye_img


def preprocess_hand(hand_raw, target_size=(640, 480)):
    """
    Preprocess hand image with center crop and resize

    Args:
        hand_raw: Raw hand image
        target_size: Target size (width, height)

    Returns:
        hand_processed: Processed hand image
    """
    h, w = hand_raw.shape[:2]
    target_w, target_h = target_size

    # Calculate aspect ratios
    aspect_ratio = target_w / target_h
    current_ratio = w / h

    # Center crop to match aspect ratio
    if current_ratio > aspect_ratio:
        # Image is wider: crop width
        new_w = int(h * aspect_ratio)
        start_x = (w - new_w) // 2
        cropped = hand_raw[:, start_x:start_x + new_w]
    else:
        # Image is taller: crop height
        new_h = int(w / aspect_ratio)
        start_y = (h - new_h) // 2
        cropped = hand_raw[start_y:start_y + new_h, :]

    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def preprocess_eye(eye_raw, target_size=(120, 90)):
    """
    Preprocess eye image: crop and resize

    Args:
        eye_raw: Raw eye image
        target_size: Target size (width, height)

    Returns:
        eye_processed: Processed eye image
    """
    h, w = eye_raw.shape[:2]
    target_w, target_h = target_size

    # Crop to remove hair (top 10% of image)
    crop_top = int(h * 0.1)
    cropped = eye_raw[crop_top:, :]

    # Center crop to match aspect ratio
    h, w = cropped.shape[:2]
    aspect_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > aspect_ratio:
        # Crop width
        new_w = int(h * aspect_ratio)
        start_x = (w - new_w) // 2
        cropped = cropped[:, start_x:start_x + new_w]
    else:
        # Crop height
        new_h = int(w / aspect_ratio)
        start_y = (h - new_h) // 2
        cropped = cropped[start_y:start_y + new_h, :]

    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def place_on_canvas(image, canvas_size=(480, 640), position=(240, 240)):
    """
    Place image on black canvas at specified position

    Args:
        image: Image to place
        canvas_size: Canvas size (height, width)
        position: Position (row, col) for center of image

    Returns:
        canvas: Canvas with image placed
    """
    canvas_h, canvas_w = canvas_size
    img_h, img_w = image.shape[:2]

    # Create black canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=image.dtype)

    # Calculate top-left corner position
    row, col = position
    start_row = row - img_h // 2
    start_col = col - img_w // 2

    # Ensure within bounds
    start_row = max(0, min(start_row, canvas_h - img_h))
    start_col = max(0, min(start_col, canvas_w - img_w))

    # Place image
    canvas[start_row:start_row + img_h, start_col:start_col + img_w] = image

    return canvas


def create_mask(shape=(480, 640), center=(240, 240), axes=(60, 45), blur_kernel=31, output_dir=None):
    """
    Create elliptical mask with Gaussian blur

    Args:
        shape: Mask shape (height, width)
        center: Ellipse center (row, col)
        axes: Ellipse axes (axis_x, axis_y)
        blur_kernel: Gaussian blur kernel size (must be odd)
        output_dir: Optional output directory to save mask

    Returns:
        mask: Blurred mask with values in [0, 1]
    """
    h, w = shape
    center_row, center_col = center
    axis_x, axis_y = axes

    # Create mask: all zeros
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw filled ellipse (255 = white)
    # OpenCV uses (x, y) format, so we need (col, row)
    cv2.ellipse(mask, (center_col, center_row), (axis_x, axis_y),
                0, 0, 360, 255, -1)

    # Save mask before blur if requested
    if output_dir:
        import os
        preprocessed_dir = os.path.join(output_dir, 'preprocessed')
        save_image(mask, os.path.join(preprocessed_dir, 'mask_ellipse.png'))

    # Apply Gaussian blur for smooth transition
    if blur_kernel > 0:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

    # Normalize to [0, 1]
    mask = mask.astype(np.float32) / 255.0

    # Add channel dimension for broadcasting
    mask = mask[:, :, np.newaxis]

    # Save blurred mask if requested
    if output_dir:
        import os
        preprocessed_dir = os.path.join(output_dir, 'preprocessed')
        save_image(mask, os.path.join(preprocessed_dir, 'mask.png'))

    return mask
