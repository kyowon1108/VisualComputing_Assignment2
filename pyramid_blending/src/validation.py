"""
Pyramid Blending Validation & Verification Module

이 모듈은 Pyramid Blending 알고리즘의 정확성을 검증합니다.

검증 프로세스:
- Phase 1: Gaussian Pyramid 정확성 검증
- Phase 2: Laplacian Pyramid 재구성 검증
- Phase 3: Blending 프로세스 검증
- Phase 4: Reconstruction 단계별 검증
- Phase 5: 최종 결과 품질 검증
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from skimage.metrics import structural_similarity


# ============================================================================
# Phase 1: Gaussian Pyramid Verification
# ============================================================================

def verify_gaussian_pyramid(hand_gp, eye_gp, mask_gp, levels=6):
    """
    Gaussian Pyramid 정확성 검증

    검증 항목:
    1. 크기 정확성: 각 레벨이 정확히 1/2씩 감소하는가?
    2. 정보 연속성: 레벨 간 smooth transition?
    3. 범위 검증: 값이 정상 범위 [0, 1]?
    4. 분산 감소: 레벨이 올라갈수록 분산 감소?
    5. Artifact 검사: 예상치 못한 이상 패턴?

    Args:
        hand_gp: Hand Gaussian pyramid
        eye_gp: Eye Gaussian pyramid
        mask_gp: Mask Gaussian pyramid
        levels: Number of pyramid levels

    Returns:
        checks: Dictionary with validation results
    """
    checks = {
        'title': 'Gaussian Pyramid Verification',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Expected sizes for 640×480 input
    expected_sizes = {
        0: (480, 640),
        1: (240, 320),
        2: (120, 160),
        3: (60, 80),
        4: (30, 40),
        5: (15, 20),
    }

    # Check 1: Size Validation
    print("\n[Phase 1.1] Gaussian Pyramid 크기 검증")
    size_checks = {}
    all_sizes_valid = True

    for level in range(min(levels, len(expected_sizes))):
        if level >= len(hand_gp):
            break

        actual_size = hand_gp[level].shape[:2]
        expected_size = expected_sizes.get(level, None)

        if expected_size:
            valid = actual_size == expected_size
            all_sizes_valid = all_sizes_valid and valid

            size_checks[f'level_{level}'] = {
                'expected': expected_size,
                'actual': actual_size,
                'valid': valid,
                'status': '✓' if valid else '✗'
            }

            print(f"  Level {level}: {actual_size} {'✓' if valid else '✗ Expected: ' + str(expected_size)}")

    checks['checks']['size_validation'] = {
        'all_valid': all_sizes_valid,
        'details': size_checks
    }

    # Check 2: Value Range
    print("\n[Phase 1.2] 값 범위 검증 (Hand Pyramid)")
    range_checks = {}
    all_ranges_valid = True

    for level in range(min(levels, len(hand_gp))):
        img = hand_gp[level]
        h_min, h_max = float(img.min()), float(img.max())
        valid = 0 <= h_min and h_max <= 1.0
        all_ranges_valid = all_ranges_valid and valid

        range_checks[f'level_{level}'] = {
            'min': h_min,
            'max': h_max,
            'valid': valid,
            'status': '✓' if valid else '✗'
        }

        print(f"  Level {level}: [{h_min:.4f}, {h_max:.4f}] {'✓' if valid else '✗'}")

    checks['checks']['range_validation'] = {
        'all_valid': all_ranges_valid,
        'details': range_checks
    }

    # Check 3: Monotonic Decrease in Variance
    print("\n[Phase 1.3] 분산 감소 검증")
    variances = [float(hand_gp[i].var()) for i in range(min(levels, len(hand_gp)))]
    monotonic = all(variances[i] >= variances[i+1] for i in range(len(variances)-1))

    checks['checks']['variance_monotonic'] = {
        'valid': monotonic,
        'variances': variances,
        'status': '✓' if monotonic else '✗'
    }

    print(f"  Variances: {[f'{v:.6f}' for v in variances]}")
    print(f"  Monotonic decrease: {'✓' if monotonic else '✗'}")

    # Check 4: Edge Energy (Blur Effect Validation)
    print("\n[Phase 1.4] 블러 효과 검증 (Edge Energy)")
    edge_energies = []

    for level in range(min(levels, len(hand_gp))):
        img = hand_gp[level]
        # Convert to uint8 for Laplacian
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Handle grayscale vs color
        if len(img_uint8.shape) == 3:
            img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_uint8

        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        edge_energy = float(np.sum(np.abs(laplacian)))
        edge_energies.append(edge_energy)

        print(f"  Level {level}: Energy = {edge_energy:.2f}")

    checks['checks']['edge_energy'] = {
        'energies': edge_energies,
        'decreasing': all(edge_energies[i] >= edge_energies[i+1] for i in range(len(edge_energies)-1)),
        'explanation': 'Edge energy should decrease with level (more blur)'
    }

    # Overall validation
    checks['overall_valid'] = all_sizes_valid and all_ranges_valid and monotonic

    return checks


def verify_opencv_vs_raw_equivalence(image, gaussian_pyramid_opencv,
                                     gaussian_pyramid_raw, levels=6):
    """
    OpenCV vs Raw Convolution 일치성 검증

    두 방식이 동일한 결과를 생성하는지 확인
    Expected: SSIM > 0.95 (거의 동일)

    Args:
        image: Original image
        gaussian_pyramid_opencv: OpenCV implementation function
        gaussian_pyramid_raw: Raw implementation function
        levels: Number of levels

    Returns:
        differences: Dictionary with comparison results
    """
    print("\n[Phase 1.5] OpenCV vs Raw 구현 비교")

    # Generate pyramids
    gp_opencv, _ = gaussian_pyramid_opencv(image, levels)
    gp_raw, _ = gaussian_pyramid_raw(image, levels)

    differences = {
        'title': 'OpenCV vs Raw Implementation Comparison',
        'timestamp': datetime.now().isoformat(),
        'levels': {}
    }

    all_valid = True

    for level in range(min(levels, len(gp_opencv), len(gp_raw))):
        img_cv = gp_opencv[level]
        img_raw = gp_raw[level]

        # Ensure same size
        if img_cv.shape != img_raw.shape:
            img_raw = cv2.resize(img_raw, (img_cv.shape[1], img_cv.shape[0]))

        # MSE
        mse = float(np.mean((img_cv - img_raw) ** 2))

        # SSIM
        if len(img_cv.shape) == 3:
            # Multi-channel SSIM
            ssim = structural_similarity(img_cv, img_raw,
                                        data_range=1.0, channel_axis=2)
        else:
            ssim = structural_similarity(img_cv, img_raw, data_range=1.0)

        max_diff = float(np.max(np.abs(img_cv - img_raw)))
        valid = ssim > 0.95
        all_valid = all_valid and valid

        differences['levels'][f'level_{level}'] = {
            'mse': mse,
            'ssim': float(ssim),
            'max_pixel_diff': max_diff,
            'valid': valid,
            'expected_ssim': 0.95,
            'status': '✓' if valid else '✗'
        }

        print(f"  Level {level}: SSIM = {ssim:.4f}, MSE = {mse:.6f} {'✓' if valid else '✗'}")

    differences['all_valid'] = all_valid

    if not all_valid:
        print("\n  ⚠️ 경고: OpenCV와 Raw 구현이 크게 다름")
        print("  가능한 원인:")
        print("    1. Kernel 정의 오류")
        print("    2. Normalization 차이")
        print("    3. Boundary handling 차이")

    return differences


# ============================================================================
# Phase 2: Laplacian Pyramid Verification
# ============================================================================

def verify_laplacian_reconstruction_accuracy(image, gaussian_pyramid_func,
                                            laplacian_pyramid_func,
                                            reconstruct_func, levels=6):
    """
    Laplacian Pyramid 재구성 정확성 검증

    정보 손실 없음 원칙:
    Original → GP → LP → Reconstruction → Original'
    Expected: PSNR > 40 dB (무시할 수준의 오차)

    Args:
        image: Original image
        gaussian_pyramid_func: Function to generate Gaussian pyramid
        laplacian_pyramid_func: Function to generate Laplacian pyramid
        reconstruct_func: Function to reconstruct from Laplacian pyramid
        levels: Number of levels

    Returns:
        reconstruction_error: Dictionary with error metrics
    """
    print("\n[Phase 2.1] Laplacian 재구성 정확성 검증")

    original = image.copy()

    # Generate pyramids
    gp, _ = gaussian_pyramid_func(original, levels)
    lp = laplacian_pyramid_func(gp)

    # Reconstruct
    reconstructed = reconstruct_func(lp, original.shape[:2])

    # Ensure same size
    if reconstructed.shape != original.shape:
        reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))

    # Calculate errors
    mse = float(np.mean((original - reconstructed) ** 2))

    # PSNR calculation
    if mse > 0:
        psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
    else:
        psnr = float('inf')

    max_error = float(np.max(np.abs(original - reconstructed)))
    mean_error = float(np.mean(np.abs(original - reconstructed)))

    valid = psnr > 40 or psnr == float('inf')

    reconstruction_error = {
        'title': 'Laplacian Reconstruction Accuracy',
        'timestamp': datetime.now().isoformat(),
        'mse': mse,
        'psnr': psnr,
        'max_error': max_error,
        'mean_error': mean_error,
        'valid': valid,
        'expected_psnr': 40.0,
        'status': '✓' if valid else '✗'
    }

    print(f"  MSE: {mse:.8f}")
    print(f"  PSNR: {psnr:.2f} dB {'✓' if valid else '✗'}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  Mean Error: {mean_error:.6f}")

    if not valid:
        print("\n  ⚠️ 경고: Reconstruction PSNR이 너무 낮음")
        print(f"  Expected: > 40 dB, Actual: {psnr:.2f} dB")
        print("  문제: 정보 손실 발생")
        reconstruction_error['warning'] = (
            f"Reconstruction PSNR too low: {psnr:.2f} dB\n"
            f"Expected: > 40 dB\n"
            f"Problem: Information loss in pyramid operations"
        )

    return reconstruction_error


def verify_laplacian_properties(hand_lp, eye_lp, levels=6):
    """
    Laplacian Pyramid 특성 검증

    Expected Properties:
    1. 대부분의 값이 0에 가까움 (detail = sparse)
    2. 크기가 감소할수록 더 sparse해짐
    3. 고주파 정보만 포함 (저주파 제외, mean ≈ 0)

    Args:
        hand_lp: Hand Laplacian pyramid
        eye_lp: Eye Laplacian pyramid
        levels: Number of levels

    Returns:
        properties: Dictionary with property checks
    """
    print("\n[Phase 2.2] Laplacian Pyramid 특성 검증")

    properties = {
        'title': 'Laplacian Pyramid Properties',
        'timestamp': datetime.now().isoformat(),
        'levels': {}
    }

    for level in range(min(levels, len(hand_lp))):
        lap = hand_lp[level]

        # Property 1: Sparsity (얼마나 0에 가까운가?)
        zero_percentage = float((np.abs(lap) < 0.01).sum() / lap.size * 100)

        # Property 2: Value Distribution
        mean_val = float(np.mean(lap))
        std_val = float(np.std(lap))
        min_val = float(np.min(lap))
        max_val = float(np.max(lap))

        # Validation
        is_sparse = zero_percentage > 30  # At least 30% near zero
        is_zero_centered = abs(mean_val) < 0.05
        has_detail = std_val > 0.001

        properties['levels'][f'level_{level}'] = {
            'zero_percentage': zero_percentage,
            'mean': mean_val,
            'std': std_val,
            'shape': lap.shape,
            'min': min_val,
            'max': max_val,
            'properties_valid': {
                'is_sparse': is_sparse,
                'is_zero_centered': is_zero_centered,
                'has_detail': has_detail
            },
            'all_valid': is_sparse and is_zero_centered and has_detail
        }

        status = '✓' if (is_sparse and is_zero_centered and has_detail) else '✗'
        print(f"  Level {level}: Sparse={zero_percentage:.1f}%, Mean={mean_val:.4f}, "
              f"Std={std_val:.4f} {status}")

    return properties


# ============================================================================
# Phase 3: Blending Process Verification
# ============================================================================

def verify_mask_properties(mask, eye_position, eye_size):
    """
    Mask 특성 검증

    검증 항목:
    1. 중심 위치 정확성
    2. 타원형 축 정확성
    3. Feathering (blur) 품질
    4. 경계 부드러움

    Args:
        mask: Generated mask
        eye_position: Expected eye position (x, y)
        eye_size: Expected eye size (semi-axes)

    Returns:
        mask_analysis: Dictionary with mask validation results
    """
    print("\n[Phase 3.1] Mask 특성 검증")

    mask_analysis = {
        'title': 'Mask Properties Verification',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Convert to 2D if needed
    if len(mask.shape) == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    # Check 1: Center accuracy
    mask_binary = (mask_2d > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours[0]) >= 5:
        ellipse = cv2.fitEllipse(contours[0])
        center, axes, angle = ellipse

        center_error = float(np.linalg.norm(
            np.array(center) - np.array(eye_position)
        ))
        center_valid = center_error < 10  # 10px tolerance

        mask_analysis['checks']['center'] = {
            'expected': eye_position,
            'actual': tuple(map(float, center)),
            'error_pixels': center_error,
            'valid': center_valid,
            'status': '✓' if center_valid else '✗'
        }

        print(f"  중심 오차: {center_error:.2f} pixels {'✓' if center_valid else '✗'}")

        # Check 2: Axes
        axes_error = [abs(axes[i] - eye_size[i]) for i in range(2)]
        axes_valid = all(err < 10 for err in axes_error)

        mask_analysis['checks']['axes'] = {
            'expected': eye_size,
            'actual': tuple(map(float, axes)),
            'valid': axes_valid,
            'status': '✓' if axes_valid else '✗'
        }

        print(f"  축 크기: {axes} {'✓' if axes_valid else '✗'}")
    else:
        mask_analysis['checks']['center'] = {
            'error': 'Could not fit ellipse to mask',
            'valid': False,
            'status': '✗'
        }

    # Check 3: Feathering quality (gradient smoothness)
    gy, gx = np.gradient(mask_2d)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    max_gradient = float(np.max(gradient_magnitude))
    mean_gradient = float(np.mean(gradient_magnitude))

    smooth = max_gradient < 0.1

    mask_analysis['checks']['feathering'] = {
        'max_gradient': max_gradient,
        'mean_gradient': mean_gradient,
        'quality': 'smooth' if smooth else 'sharp',
        'valid': smooth,
        'status': '✓' if smooth else '✗'
    }

    print(f"  Feathering: {'Smooth ✓' if smooth else 'Sharp ✗'} "
          f"(max grad={max_gradient:.4f})")

    # Check 4: Value range
    min_val = float(np.min(mask_2d))
    max_val = float(np.max(mask_2d))
    range_valid = min_val >= 0 and max_val <= 1.0

    mask_analysis['checks']['value_range'] = {
        'min': min_val,
        'max': max_val,
        'valid': range_valid,
        'status': '✓' if range_valid else '✗'
    }

    print(f"  값 범위: [{min_val:.4f}, {max_val:.4f}] {'✓' if range_valid else '✗'}")

    return mask_analysis


def verify_level_blending(hand_lap, eye_lap, mask_gp, blended_lap, level):
    """
    각 레벨에서의 blending 정확성 검증

    Formula Check:
    blended[i] = hand_lap[i] * (1 - mask_gp[i]) + eye_lap[i] * mask_gp[i]

    Args:
        hand_lap: Hand Laplacian pyramid
        eye_lap: Eye Laplacian pyramid
        mask_gp: Mask Gaussian pyramid
        blended_lap: Blended Laplacian pyramid
        level: Level to check

    Returns:
        blend_error: Dictionary with blending verification results
    """
    h = hand_lap[level]
    e = eye_lap[level]
    m = mask_gp[level]
    b = blended_lap[level]

    # Ensure mask has correct dimensions
    if len(h.shape) == 2 and len(m.shape) == 3:
        m = m[:, :, 0]
    elif len(h.shape) == 3 and len(m.shape) == 2:
        m = m[:, :, np.newaxis]

    # Ensure same size
    if e.shape != h.shape:
        e = cv2.resize(e, (h.shape[1], h.shape[0]))
    if m.shape[:2] != h.shape[:2]:
        m = cv2.resize(m, (h.shape[1], h.shape[0]))
        if len(h.shape) == 3 and len(m.shape) == 2:
            m = m[:, :, np.newaxis]

    # Manual calculation
    expected_blend = h * (1 - m) + e * m

    # Compare
    mse = float(np.mean((b - expected_blend) ** 2))
    max_diff = float(np.max(np.abs(b - expected_blend)))
    valid = np.allclose(b, expected_blend, atol=1e-4)

    blend_error = {
        'level': level,
        'mse': mse,
        'max_diff': max_diff,
        'valid': valid,
        'formula_correct': valid,
        'status': '✓' if valid else '✗'
    }

    return blend_error


# ============================================================================
# Phase 4: Reconstruction Quality Verification
# ============================================================================

def verify_reconstruction_steps(blended_lap, reconstruct_func, levels=6):
    """
    Reconstruction 각 단계의 정확성 검증

    Step 1: result = blended_lap[5]
    Step 2: result = pyrUp(result) + blended_lap[4]
    Step 3: result = pyrUp(result) + blended_lap[3]
    ...
    Step 6: result = pyrUp(result) + blended_lap[0]

    각 단계에서:
    1. pyrUp 크기 정확성 확인
    2. 덧셈 전에 크기 일치 확인
    3. 값의 범위 유지 확인

    Args:
        blended_lap: Blended Laplacian pyramid
        reconstruct_func: Reconstruction function
        levels: Number of levels

    Returns:
        reconstruction_log: Dictionary with step-by-step validation
    """
    print("\n[Phase 4] Reconstruction 단계별 검증")

    reconstruction_log = {
        'title': 'Reconstruction Step-by-Step Verification',
        'timestamp': datetime.now().isoformat(),
        'steps': {}
    }

    # Start from smallest level
    result = blended_lap[-1].copy()
    reconstruction_log['steps']['step_0_base'] = {
        'description': 'Base level (smallest)',
        'shape': result.shape,
        'min': float(result.min()),
        'max': float(result.max()),
        'mean': float(result.mean())
    }

    print(f"  Step 0 (Base): Shape {result.shape}, "
          f"Range [{result.min():.4f}, {result.max():.4f}]")

    all_steps_valid = True

    for step in range(len(blended_lap) - 2, -1, -1):
        step_num = len(blended_lap) - 1 - step

        # Upsample
        result_up = cv2.pyrUp(result)

        # Size check before addition
        expected_size = blended_lap[step].shape[:2]
        actual_size = result_up.shape[:2]

        size_match = expected_size == actual_size

        if not size_match:
            all_steps_valid = False
            reconstruction_log['steps'][f'step_{step_num}_ERROR'] = {
                'error': 'Size mismatch after pyrUp',
                'expected': expected_size,
                'actual': actual_size,
                'recovery': 'Using cv2.resize to fix'
            }
            # Recovery: resize to match
            result_up = cv2.resize(result_up, (expected_size[1], expected_size[0]))
            print(f"  Step {step_num}: ✗ Size mismatch (recovered)")
        else:
            print(f"  Step {step_num}: ✓ Size match")

        # Addition
        result = result_up + blended_lap[step]

        # Check value range
        value_range_ok = result.min() >= -0.5 and result.max() <= 1.5

        reconstruction_log['steps'][f'step_{step_num}'] = {
            'after_pyrUp': {
                'shape': result_up.shape,
                'min': float(result_up.min()),
                'max': float(result_up.max())
            },
            'after_addition': {
                'shape': result.shape,
                'min': float(result.min()),
                'max': float(result.max()),
                'mean': float(result.mean()),
                'value_range_ok': value_range_ok
            },
            'size_match': size_match,
            'valid': size_match and value_range_ok
        }

    reconstruction_log['all_steps_valid'] = all_steps_valid

    print(f"  최종 Shape: {result.shape}")
    print(f"  모든 단계 유효: {'✓' if all_steps_valid else '✗'}")

    return reconstruction_log


# ============================================================================
# Phase 5: Final Result Validation
# ============================================================================

def verify_final_result_quality(result, hand_original, eye_original, mask):
    """
    최종 결과 이미지 품질 검증

    검증 항목:
    1. 크기: 손 이미지와 동일 (640×480)
    2. 값 범위: [0, 1]
    3. 손 영역: 손의 특성 유지
    4. 눈 영역: 눈의 특성 명확
    5. 경계 영역: Smooth transition
    6. 색상: 자연스러운 blend

    Args:
        result: Final blended result
        hand_original: Original hand image
        eye_original: Original eye image
        mask: Blending mask

    Returns:
        quality_report: Dictionary with quality metrics
    """
    print("\n[Phase 5] 최종 결과 품질 검증")

    quality_report = {
        'title': 'Final Result Quality Verification',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Check 1: Size
    expected_shape = (480, 640, 3)
    size_valid = result.shape == expected_shape

    quality_report['checks']['size'] = {
        'expected': expected_shape,
        'actual': result.shape,
        'valid': size_valid,
        'status': '✓' if size_valid else '✗'
    }

    print(f"  크기: {result.shape} {'✓' if size_valid else '✗'}")

    # Check 2: Value range
    min_val = float(result.min())
    max_val = float(result.max())
    range_valid = min_val >= 0 and max_val <= 1.0
    clipping_needed = min_val < 0 or max_val > 1.0

    quality_report['checks']['value_range'] = {
        'min': min_val,
        'max': max_val,
        'valid': range_valid,
        'clipping_needed': clipping_needed,
        'status': '✓' if range_valid else '✗'
    }

    print(f"  값 범위: [{min_val:.4f}, {max_val:.4f}] {'✓' if range_valid else '✗'}")

    # Convert to 2D mask if needed
    if len(mask.shape) == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    # Ensure mask matches result size
    if mask_2d.shape[:2] != result.shape[:2]:
        mask_2d = cv2.resize(mask_2d, (result.shape[1], result.shape[0]))

    # Check 3: Hand region analysis (mask < 0.3)
    hand_mask = mask_2d < 0.3

    if hand_mask.sum() > 0 and hand_original.shape == result.shape:
        hand_region_result = result[hand_mask]
        hand_region_original = hand_original[hand_mask]

        # Convert to uint8 for SSIM
        hand_result_uint8 = (np.clip(hand_region_result, 0, 1) * 255).astype(np.uint8)
        hand_orig_uint8 = (np.clip(hand_region_original, 0, 1) * 255).astype(np.uint8)

        # Calculate similarity
        if hand_result_uint8.size > 0 and hand_orig_uint8.size > 0:
            # MSE instead of SSIM for regions
            hand_mse = float(np.mean((hand_region_result - hand_region_original) ** 2))
            hand_similarity = 1.0 - min(hand_mse, 1.0)  # Approximate similarity
            hand_valid = hand_similarity > 0.7

            quality_report['checks']['hand_preservation'] = {
                'region_similarity': hand_similarity,
                'mse': hand_mse,
                'valid': hand_valid,
                'interpretation': 'Hand features preserved' if hand_valid else 'Hand distorted',
                'status': '✓' if hand_valid else '✗'
            }

            print(f"  손 영역 보존: Similarity={hand_similarity:.3f} {'✓' if hand_valid else '✗'}")

    # Check 4: Eye region analysis (mask > 0.7)
    eye_mask = mask_2d > 0.7

    if eye_mask.sum() > 100:  # At least 100 pixels
        eye_region_intensity = float(np.mean(result[eye_mask]))

        if eye_original.shape == result.shape:
            eye_original_intensity = float(np.mean(eye_original[eye_mask]))
            intensity_diff = float(abs(eye_region_intensity - eye_original_intensity))
            eye_valid = intensity_diff < 0.3

            quality_report['checks']['eye_visibility'] = {
                'result_intensity': eye_region_intensity,
                'original_intensity': eye_original_intensity,
                'difference': intensity_diff,
                'valid': eye_valid,
                'interpretation': 'Eye visible and clear' if eye_valid else 'Eye visibility issues',
                'status': '✓' if eye_valid else '✗'
            }

            print(f"  눈 영역 가시성: Diff={intensity_diff:.3f} {'✓' if eye_valid else '✗'}")

    # Check 5: Boundary smoothness
    boundary_mask = (mask_2d > 0.2) & (mask_2d < 0.8)  # Transition zone

    if boundary_mask.sum() > 0:
        boundary_region = result[boundary_mask]
        boundary_smoothness = float(np.std(boundary_region))
        boundary_valid = boundary_smoothness < 0.15

        quality_report['checks']['boundary_smoothness'] = {
            'std_in_transition': boundary_smoothness,
            'valid': boundary_valid,
            'interpretation': 'Smooth boundary' if boundary_valid else 'Boundary artifacts',
            'status': '✓' if boundary_valid else '✗'
        }

        print(f"  경계 부드러움: Std={boundary_smoothness:.4f} {'✓' if boundary_valid else '✗'}")

    # Check 6: Color harmony
    if len(result.shape) == 3:
        color_channels = [float(result[:, :, i].mean()) for i in range(3)]
        color_balance = float(np.std(color_channels))
        color_valid = color_balance < 0.2

        quality_report['checks']['color_balance'] = {
            'r_mean': color_channels[0],
            'g_mean': color_channels[1],
            'b_mean': color_channels[2],
            'balance_score': color_balance,
            'valid': color_valid,
            'interpretation': 'Balanced colors' if color_valid else 'Color distortion',
            'status': '✓' if color_valid else '✗'
        }

        print(f"  색상 균형: Std={color_balance:.4f} {'✓' if color_valid else '✗'}")

    return quality_report


# ============================================================================
# Report Generation
# ============================================================================

def generate_validation_report(all_checks, output_dir):
    """
    모든 검증 결과를 종합하여 최종 리포트 생성

    Output:
    1. validation_report.json - 모든 수치 데이터
    2. validation_summary.txt - 인간이 읽을 수 있는 요약

    Args:
        all_checks: Dictionary containing all validation results
        output_dir: Output directory for reports

    Returns:
        report_path: Path to generated report
    """
    print("\n" + "="*80)
    print("[Report] 검증 리포트 생성 중...")
    print("="*80)

    # Create validation directory
    validation_dir = os.path.join(output_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    # Count checks
    total_checks = 0
    passed_checks = 0
    failed_checks = 0
    warnings = []

    def count_checks(obj, prefix=''):
        nonlocal total_checks, passed_checks, failed_checks
        if isinstance(obj, dict):
            if 'valid' in obj:
                total_checks += 1
                if obj['valid']:
                    passed_checks += 1
                else:
                    failed_checks += 1
                    if 'warning' in obj:
                        warnings.append(f"{prefix}: {obj['warning']}")
            for key, value in obj.items():
                count_checks(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                count_checks(item, f"{prefix}[{i}]")

    count_checks(all_checks)

    # Determine overall status
    overall_status = 'PASS' if failed_checks == 0 else 'FAIL' if failed_checks > 5 else 'WARNING'

    # Create comprehensive report
    report = {
        'title': 'Pyramid Blending Algorithm Verification Report',
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'summary': {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'pass_rate': f"{(passed_checks/total_checks*100):.1f}%" if total_checks > 0 else "N/A",
            'warnings': warnings
        },
        'phases': all_checks
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert report to JSON-serializable format
    report_serializable = convert_to_json_serializable(report)

    # Save JSON report
    json_path = os.path.join(validation_dir, 'validation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON 리포트 저장: {json_path}")

    # Generate human-readable summary
    summary_path = os.path.join(validation_dir, 'validation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Pyramid Blending Algorithm - Validation Report\n")
        f.write("="*80 + "\n\n")

        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"전체 상태: {overall_status}\n\n")

        f.write("-"*80 + "\n")
        f.write("요약\n")
        f.write("-"*80 + "\n")
        f.write(f"총 검증 항목: {total_checks}\n")
        f.write(f"통과: {passed_checks}\n")
        f.write(f"실패: {failed_checks}\n")
        f.write(f"통과율: {(passed_checks/total_checks*100):.1f}%\n\n")

        if warnings:
            f.write("-"*80 + "\n")
            f.write("경고 사항\n")
            f.write("-"*80 + "\n")
            for i, warning in enumerate(warnings, 1):
                f.write(f"{i}. {warning}\n")
            f.write("\n")

        f.write("-"*80 + "\n")
        f.write("Phase별 상세 결과\n")
        f.write("-"*80 + "\n\n")

        # Phase summaries
        phase_names = {
            'phase_1': 'Phase 1: Gaussian Pyramid 검증',
            'phase_2': 'Phase 2: Laplacian Pyramid 검증',
            'phase_3': 'Phase 3: Blending 프로세스 검증',
            'phase_4': 'Phase 4: Reconstruction 검증',
            'phase_5': 'Phase 5: 최종 결과 품질 검증'
        }

        for phase_key, phase_name in phase_names.items():
            if phase_key in all_checks:
                f.write(f"{phase_name}\n")
                phase_data = all_checks[phase_key]

                # Extract status
                if isinstance(phase_data, dict):
                    if 'overall_valid' in phase_data:
                        status = '✓ PASS' if phase_data['overall_valid'] else '✗ FAIL'
                        f.write(f"  상태: {status}\n")
                    elif 'all_valid' in phase_data:
                        status = '✓ PASS' if phase_data['all_valid'] else '✗ FAIL'
                        f.write(f"  상태: {status}\n")
                    elif 'valid' in phase_data:
                        status = '✓ PASS' if phase_data['valid'] else '✗ FAIL'
                        f.write(f"  상태: {status}\n")

                f.write("\n")

        f.write("="*80 + "\n")

    print(f"✓ 요약 리포트 저장: {summary_path}")

    # Print summary to console
    print("\n" + "="*80)
    print("검증 결과 요약")
    print("="*80)
    print(f"전체 상태: {overall_status}")
    print(f"통과율: {passed_checks}/{total_checks} ({(passed_checks/total_checks*100):.1f}%)")
    print("="*80)

    return json_path, summary_path


def print_validation_summary(report):
    """
    검증 리포트를 콘솔에 출력

    Args:
        report: Validation report dictionary
    """
    print("\n" + "="*80)
    print("Pyramid Blending Algorithm - Validation Summary")
    print("="*80)

    summary = report['summary']
    print(f"\n전체 상태: {report['overall_status']}")
    print(f"총 검증: {summary['total_checks']} | "
          f"통과: {summary['passed_checks']} | "
          f"실패: {summary['failed_checks']}")
    print(f"통과율: {summary['pass_rate']}")

    if summary['warnings']:
        print(f"\n⚠️ 경고: {len(summary['warnings'])}개")
        for warning in summary['warnings'][:3]:  # Show first 3
            print(f"  - {warning}")

    print("="*80)
