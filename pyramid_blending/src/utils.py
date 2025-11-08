"""
Utility functions for image pyramid blending project
"""
import os
import time
import cv2
import numpy as np
from functools import wraps
from datetime import datetime


def create_output_directories():
    """Create all necessary output directories"""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

    directories = [
        'preprocessed',
        'pyramids/hand_gaussian',
        'pyramids/eye_gaussian',
        'pyramids/mask_gaussian',
        'pyramids/hand_laplacian',
        'pyramids/eye_laplacian',
        'blending_results',
        'visualization',
        'reports'
    ]

    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

    return base_dir


def save_image(image, path):
    """
    Save image with proper normalization

    Args:
        image: Image array (can be float32 [0,1] or uint8 [0,255])
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Normalize to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Clip to [0, 1] range and convert to [0, 255]
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    # Handle single channel images
    if len(image.shape) == 2:
        cv2.imwrite(path, image)
    else:
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_image(path, normalize=True):
    """
    Load image and optionally normalize

    Args:
        path: Image path
        normalize: If True, normalize to [0, 1] float32

    Returns:
        image: Loaded image in RGB format
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image (BGR format)
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


def normalize_image(image):
    """
    Normalize image to [0, 1] float32

    Args:
        image: Image array

    Returns:
        normalized: Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.float32 or image.dtype == np.float64:
        return np.clip(image, 0, 1).astype(np.float32)
    else:
        return image.astype(np.float32)


def measure_time(func):
    """
    Decorator to measure function execution time

    Usage:
        @measure_time
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, elapsed_time
    return wrapper


def log_message(message, level="INFO", log_file=None):
    """
    Log message with timestamp

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR)
        log_file: Optional file path to write log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"[{timestamp}] [{level}] {message}"

    print(log_text)

    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_text + '\n')


def get_pyramid_shape_str(pyramid):
    """
    Get string representation of pyramid shapes

    Args:
        pyramid: List of images

    Returns:
        shape_str: String describing all levels
    """
    shapes = [f"Level {i}: {img.shape}" for i, img in enumerate(pyramid)]
    return ", ".join(shapes)


def calculate_memory_usage(pyramid):
    """
    Calculate total memory usage of pyramid in MB

    Args:
        pyramid: List of images

    Returns:
        memory_mb: Memory usage in megabytes
    """
    total_bytes = sum(img.nbytes for img in pyramid)
    return total_bytes / (1024 * 1024)


def ensure_same_size(img1, img2):
    """
    Ensure two images have the same size by resizing img2 to match img1

    Args:
        img1: Reference image
        img2: Image to resize

    Returns:
        img2_resized: Resized img2
    """
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img2


def gaussian_kernel_5x5():
    """
    Create 5x5 Gaussian kernel for pyramid generation

    Returns:
        kernel: 5x5 normalized Gaussian kernel
    """
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32)

    return kernel / 256.0


def convolve2d(image, kernel):
    """
    Apply 2D convolution to image

    Args:
        image: Input image (can be multi-channel)
        kernel: Convolution kernel

    Returns:
        result: Convolved image
    """
    if len(image.shape) == 2:
        # Single channel
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    else:
        # Multi-channel: apply to each channel
        channels = [cv2.filter2D(image[:, :, i], -1, kernel, borderType=cv2.BORDER_REFLECT)
                   for i in range(image.shape[2])]
        return np.stack(channels, axis=2)


def upsample_raw(image, kernel):
    """
    Upsample image by 2x using raw implementation (inverse of downsampling)

    This is the raw implementation of cv2.pyrUp():
    1. Create 2x size image with zeros
    2. Place original pixels at even positions
    3. Apply Gaussian convolution for interpolation
    4. Multiply by 4 to normalize (compensate for zero insertion)

    Args:
        image: Input image to upsample
        kernel: Gaussian kernel for interpolation

    Returns:
        upsampled: 2x upsampled image
    """
    h, w = image.shape[:2]

    # Create output image with 2x size
    new_h, new_w = h * 2, w * 2

    if len(image.shape) == 2:
        # Single channel
        upsampled = np.zeros((new_h, new_w), dtype=image.dtype)
        # Place original pixels at even positions (0, 2, 4, ...)
        upsampled[::2, ::2] = image
    else:
        # Multi-channel
        upsampled = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        # Place original pixels at even positions
        upsampled[::2, ::2, :] = image

    # Apply Gaussian convolution for interpolation
    upsampled = convolve2d(upsampled, kernel)

    # Multiply by 4 to compensate for zero insertion
    # This is because we inserted 3 zeros for every 1 pixel (4x total pixels)
    upsampled = upsampled * 4.0

    return upsampled


def create_log_file(output_dir):
    """
    Create a new log file

    Args:
        output_dir: Output directory

    Returns:
        log_path: Path to log file
    """
    log_path = os.path.join(output_dir, 'reports', 'processing_log.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Clear existing log
    with open(log_path, 'w') as f:
        f.write(f"Image Pyramid Blending Processing Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    return log_path
