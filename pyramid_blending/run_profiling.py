"""
Performance Profiling Script

처리 속도, 메모리 사용량 측정
"""
import os
import time
import psutil
import numpy as np
from src.preprocessing import load_and_preprocess, create_mask
from src.pyramid_generation import gaussian_pyramid_opencv, gaussian_pyramid_raw, laplacian_pyramid
from src.blending import pyramid_blending, lab_blending, yuv_blending, direct_blending
from src.reconstruction import reconstruct_from_laplacian, blend_pyramids_at_level
import json


def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def profile_operation(func, *args, **kwargs):
    """
    Profile a function's execution time and memory

    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to function

    Returns:
        result, time_ms, memory_mb
    """
    # Measure memory before
    mem_before = measure_memory()

    # Measure time
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    # Measure memory after
    mem_after = measure_memory()

    time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    memory_mb = mem_after - mem_before

    return result, time_ms, memory_mb


def run_profiling():
    """Run comprehensive profiling"""
    print("\n" + "="*80)
    print(" "*20 + "PERFORMANCE PROFILING")
    print("="*80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output', 'profiling')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. Load images
    print("\n[Phase 1] Image Loading & Preprocessing")
    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')

    (hand_img, eye_img), time_ms, mem_mb = profile_operation(
        load_and_preprocess, hand_path, eye_path
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['preprocessing'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 2. Mask creation
    print("\n[Phase 2] Mask Creation")
    mask, time_ms, mem_mb = profile_operation(
        create_mask, (480, 640), (325, 315), (48, 36), 31, None
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['mask_creation'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 3. Gaussian Pyramid (OpenCV)
    print("\n[Phase 3] Gaussian Pyramid Generation (OpenCV)")
    levels = 6
    (hand_gp, _), time_ms, mem_mb = profile_operation(
        gaussian_pyramid_opencv, hand_img, levels
    )
    print(f"  Time: {time_ms:.2f} ms ({time_ms/levels:.2f} ms per level)")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['gaussian_opencv'] = {'time_ms': time_ms, 'memory_mb': mem_mb, 'time_per_level': time_ms/levels}

    # 4. Gaussian Pyramid (Raw)
    print("\n[Phase 4] Gaussian Pyramid Generation (Raw)")
    (hand_gp_raw, _), time_ms, mem_mb = profile_operation(
        gaussian_pyramid_raw, hand_img, levels
    )
    print(f"  Time: {time_ms:.2f} ms ({time_ms/levels:.2f} ms per level)")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['gaussian_raw'] = {'time_ms': time_ms, 'memory_mb': mem_mb, 'time_per_level': time_ms/levels}

    # Compare OpenCV vs Raw
    speedup = time_ms / results['gaussian_opencv']['time_ms']
    print(f"  Raw is {speedup:.2f}x slower than OpenCV")

    # 5. Laplacian Pyramid
    print("\n[Phase 5] Laplacian Pyramid Generation")
    hand_lap, time_ms, mem_mb = profile_operation(
        laplacian_pyramid, hand_gp
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['laplacian'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # Prepare for blending
    eye_gp, _ = gaussian_pyramid_opencv(eye_img, levels)
    mask_gp, _ = gaussian_pyramid_opencv(mask, levels)
    eye_lap = laplacian_pyramid(eye_gp)

    # 6. Direct Blending
    print("\n[Phase 6] Direct Blending")
    direct_result, time_ms, mem_mb = profile_operation(
        direct_blending, hand_img, eye_img, mask
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['direct_blending'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 7. Pyramid Blending (RGB)
    print("\n[Phase 7] Pyramid Blending (RGB, 5-level)")
    blended_lap = blend_pyramids_at_level(hand_lap, eye_lap, mask_gp, 5)
    rgb_result, time_ms, mem_mb = profile_operation(
        reconstruct_from_laplacian, blended_lap, (480, 640), False, 0
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['pyramid_blending_rgb'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 8. LAB Blending
    print("\n[Phase 8] LAB Color Space Blending (5-level)")
    lab_result, time_ms, mem_mb = profile_operation(
        lab_blending, hand_lap, eye_lap, mask_gp, hand_img, eye_img, 5
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['pyramid_blending_lab'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 9. YUV Blending
    print("\n[Phase 9] YUV Color Space Blending (5-level)")
    yuv_result, time_ms, mem_mb = profile_operation(
        yuv_blending, hand_lap, eye_lap, mask_gp, hand_img, eye_img, 5
    )
    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Memory: {mem_mb:.2f} MB")
    results['pyramid_blending_yuv'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # 10. Different pyramid levels
    print("\n[Phase 10] Pyramid Blending at Different Levels")
    for level in [0, 1, 3, 5]:
        blended_lap_full = blend_pyramids_at_level(hand_lap, eye_lap, mask_gp, None)
        result, time_ms, mem_mb = profile_operation(
            reconstruct_from_laplacian, blended_lap_full, (480, 640), False, level
        )
        print(f"  Level {level}: {time_ms:.2f} ms, {mem_mb:.2f} MB")
        results[f'pyramid_level_{level}'] = {'time_ms': time_ms, 'memory_mb': mem_mb}

    # Save results
    print("\n" + "="*80)
    print(" "*25 + "SAVING RESULTS")
    print("="*80)

    results_path = os.path.join(output_dir, 'profiling_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Print summary table
    print("\n" + "="*80)
    print(" "*25 + "SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Operation':<35} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 80)

    operation_names = {
        'preprocessing': 'Image Preprocessing',
        'mask_creation': 'Mask Creation',
        'gaussian_opencv': 'Gaussian Pyramid (OpenCV)',
        'gaussian_raw': 'Gaussian Pyramid (Raw)',
        'laplacian': 'Laplacian Pyramid',
        'direct_blending': 'Direct Blending',
        'pyramid_blending_rgb': 'Pyramid Blending (RGB)',
        'pyramid_blending_lab': 'Pyramid Blending (LAB)',
        'pyramid_blending_yuv': 'Pyramid Blending (YUV)',
    }

    for key, name in operation_names.items():
        if key in results:
            time_val = results[key]['time_ms']
            mem_val = results[key].get('memory_mb', 0)
            print(f"{name:<35} {time_val:<15.2f} {mem_val:<15.2f}")

    # System info
    print("\n" + "="*80)
    print(" "*25 + "SYSTEM INFO")
    print("="*80)
    print(f"  CPU Count: {psutil.cpu_count()}")
    print(f"  Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Python Process Memory: {measure_memory():.2f} MB")

    print("\n" + "="*80)
    print(" "*20 + "✓ PROFILING COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}/")
    print()

    return results


if __name__ == '__main__':
    run_profiling()
