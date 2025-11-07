"""
Quality metrics calculation module
"""
import numpy as np
import json
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


def calculate_metrics(result, reference):
    """
    Calculate quality metrics between result and reference images

    Args:
        result: Result image (H, W, 3) or (H, W)
        reference: Reference image (H, W, 3) or (H, W)

    Returns:
        metrics: Dictionary containing SSIM, MSE, and PSNR
    """
    # Ensure same size
    if result.shape != reference.shape:
        import cv2
        result = cv2.resize(result, (reference.shape[1], reference.shape[0]))

    # Ensure float32 in [0, 1]
    if result.dtype == np.uint8:
        result = result.astype(np.float32) / 255.0
    if reference.dtype == np.uint8:
        reference = reference.astype(np.float32) / 255.0

    # Calculate SSIM
    if len(result.shape) == 3:
        # Multi-channel: calculate SSIM for each channel and average
        ssim_value = ssim(reference, result, multichannel=True, channel_axis=2,
                         data_range=1.0)
    else:
        # Single channel
        ssim_value = ssim(reference, result, data_range=1.0)

    # Calculate MSE
    mse_value = mean_squared_error(reference.flatten(), result.flatten())

    # Calculate PSNR
    if mse_value > 0:
        psnr_value = 20 * np.log10(1.0 / np.sqrt(mse_value))
    else:
        psnr_value = float('inf')

    metrics = {
        'ssim': float(ssim_value),
        'mse': float(mse_value),
        'psnr': float(psnr_value)
    }

    return metrics


def calculate_all_metrics(results_dict, reference):
    """
    Calculate metrics for multiple results

    Args:
        results_dict: Dictionary mapping method names to result images
        reference: Reference image

    Returns:
        all_metrics: Dictionary mapping method names to metrics
    """
    all_metrics = {}

    for method_name, result_image in results_dict.items():
        metrics = calculate_metrics(result_image, reference)
        all_metrics[method_name] = metrics

    return all_metrics


def save_metrics_report(metrics_dict, output_path):
    """
    Save metrics report as JSON

    Args:
        metrics_dict: Dictionary of metrics
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Format metrics for better readability
    formatted_metrics = {}
    for method, metrics in metrics_dict.items():
        formatted_metrics[method] = {
            'ssim': round(metrics['ssim'], 4),
            'mse': round(metrics['mse'], 4),
            'psnr': round(metrics['psnr'], 2) if metrics['psnr'] != float('inf') else 'inf'
        }

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(formatted_metrics, f, indent=2)

    print(f"\n✓ Metrics report saved to: {output_path}")


def print_metrics_table(metrics_dict):
    """
    Print metrics in a formatted table

    Args:
        metrics_dict: Dictionary of metrics
    """
    print("\n" + "="*80)
    print("Quality Metrics Comparison")
    print("="*80)
    print(f"{'Method':<25} {'SSIM':>10} {'MSE':>12} {'PSNR':>10}")
    print("-"*80)

    for method, metrics in metrics_dict.items():
        ssim_val = metrics['ssim']
        mse_val = metrics['mse']
        psnr_val = metrics['psnr'] if metrics['psnr'] != float('inf') else 999.99

        print(f"{method:<25} {ssim_val:>10.4f} {mse_val:>12.4f} {psnr_val:>10.2f}")

    print("="*80)


def compare_methods(metrics_dict):
    """
    Compare different methods and identify best performers

    Args:
        metrics_dict: Dictionary of metrics

    Returns:
        comparison: Dictionary with best performers (None if no metrics)
    """
    # Handle empty metrics
    if not metrics_dict:
        return None

    # Find best SSIM (higher is better)
    best_ssim_method = max(metrics_dict.items(),
                          key=lambda x: x[1]['ssim'])

    # Find best MSE (lower is better)
    best_mse_method = min(metrics_dict.items(),
                         key=lambda x: x[1]['mse'])

    # Find best PSNR (higher is better)
    best_psnr_method = max(metrics_dict.items(),
                          key=lambda x: x[1]['psnr'] if x[1]['psnr'] != float('inf') else 0)

    comparison = {
        'best_ssim': {
            'method': best_ssim_method[0],
            'value': best_ssim_method[1]['ssim']
        },
        'best_mse': {
            'method': best_mse_method[0],
            'value': best_mse_method[1]['mse']
        },
        'best_psnr': {
            'method': best_psnr_method[0],
            'value': best_psnr_method[1]['psnr']
        }
    }

    return comparison


def generate_analysis_summary(metrics_dict, output_path):
    """
    Generate analysis summary text file

    Args:
        metrics_dict: Dictionary of metrics
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    comparison = compare_methods(metrics_dict)

    with open(output_path, 'w') as f:
        f.write("Image Pyramid Blending - Analysis Summary\n")
        f.write("="*80 + "\n\n")

        if comparison:
            f.write("Best Performers:\n")
            f.write(f"  - Highest SSIM: {comparison['best_ssim']['method']} "
                   f"({comparison['best_ssim']['value']:.4f})\n")
            f.write(f"  - Lowest MSE: {comparison['best_mse']['method']} "
                   f"({comparison['best_mse']['value']:.4f})\n")
            f.write(f"  - Highest PSNR: {comparison['best_psnr']['method']} "
                   f"({comparison['best_psnr']['value']:.2f})\n\n")
        else:
            f.write("No metrics available for comparison.\n\n")

        if metrics_dict:
            f.write("Detailed Metrics:\n")
            f.write("-"*80 + "\n")

            for method, metrics in sorted(metrics_dict.items()):
                f.write(f"\n{method}:\n")
                f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
                f.write(f"  MSE:  {metrics['mse']:.4f}\n")
                psnr_str = f"{metrics['psnr']:.2f}" if metrics['psnr'] != float('inf') else "inf"
                f.write(f"  PSNR: {psnr_str}\n")
        else:
            f.write("Detailed Metrics:\n")
            f.write("-"*80 + "\n")
            f.write("No metrics calculated (reference images not provided).\n")

        f.write("\n" + "="*80 + "\n")

        # Analysis
        f.write("\nAnalysis:\n")
        f.write("- Direct blending typically shows lower quality due to hard transitions\n")
        f.write("- Pyramid blending improves quality by blending at multiple scales\n")
        f.write("- 5-level pyramid usually provides optimal balance\n")
        f.write("- LAB color space may help preserve color characteristics\n")

    print(f"\n✓ Analysis summary saved to: {output_path}")
