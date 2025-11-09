"""
Comprehensive Comparison Page

모든 블렌딩 방법을 한 페이지에 종합 비교
"""
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.utils import load_image


def create_comprehensive_comparison(output_dir):
    """
    Create all-in-one comparison visualization

    Args:
        output_dir: Output directory containing all results
    """
    print("\n" + "="*80)
    print(" "*15 + "COMPREHENSIVE COMPARISON PAGE")
    print("="*80)

    blending_dir = os.path.join(output_dir, 'blending_results')
    reports_dir = os.path.join(output_dir, 'reports')

    # Load metrics
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load images
    methods = {
        'direct': 'Direct Blend',
        'pyramid_to_L0': 'Pyramid to L0 (Full)',
        'pyramid_to_L3': 'Pyramid to L3',
        'pyramid_to_L5': 'Pyramid to L5 (Coarse)'
    }

    images = {}
    for key, name in methods.items():
        if key == 'direct':
            path = os.path.join(blending_dir, 'direct_blend.jpg')
        elif 'pyramid' in key:
            # pyramid_to_L0 -> pyramid_blend_to_L0.jpg
            level = key.replace('pyramid_', '')
            path = os.path.join(blending_dir, f'pyramid_blend_{level}.jpg')
        else:
            # Other methods
            path = os.path.join(blending_dir, f'{key}.jpg')

        if os.path.exists(path):
            images[key] = load_image(path)
            print(f"  ✓ Loaded: {name}")
        else:
            print(f"  ⚠ Missing: {name} at {path}")

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.25)

    # --- Row 1-2: Image comparisons ---
    for idx, (key, name) in enumerate(methods.items()):
        if key not in images:
            continue

        row = idx // 3
        col = idx % 3

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(images[key])
        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.axis('off')

        # Add metrics as text
        if key in metrics or f'{key.replace("pyramid_", "")}' in metrics:
            metric_key = key.replace('pyramid_', '')  # pyramid_to_L0 -> to_L0
            if metric_key in metrics:
                m = metrics[metric_key]
            elif key == 'direct':
                m = metrics.get('direct_blending', {})
            else:
                m = metrics.get(key, {})

            if m:
                ssim = m.get('ssim', 'N/A')
                mse = m.get('mse', 'N/A')
                psnr = m.get('psnr', 'N/A')

                if isinstance(ssim, (int, float)):
                    text = f"SSIM: {ssim:.4f}\nMSE: {mse:.5f}\nPSNR: {psnr:.2f} dB"
                else:
                    text = "Metrics: N/A"

                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Row 3: Quality metrics comparison ---
    ax = fig.add_subplot(gs[2, :2])

    # Extract metrics for comparison
    method_names = []
    ssim_values = []
    mse_values = []

    for key in methods.keys():
        metric_key = key.replace('pyramid_', '')
        if key == 'direct':
            m = metrics.get('direct_blending', {})
        elif metric_key in metrics:
            m = metrics[metric_key]
        elif key in metrics:
            m = metrics[key]
        else:
            continue

        if 'ssim' in m:
            method_names.append(methods[key])
            ssim_values.append(m['ssim'])
            mse_values.append(m['mse'])

    x = np.arange(len(method_names))
    width = 0.35

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, ssim_values, width, label='SSIM', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, mse_values, width, label='MSE', color='red', alpha=0.7)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('SSIM', fontweight='bold', color='green')
    ax2.set_ylabel('MSE', fontweight='bold', color='red')
    ax.set_title('Quality Metrics Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(axis='y', alpha=0.3)

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # --- Row 3 Col 3-4: Performance comparison ---
    profiling_path = os.path.join(output_dir, 'profiling', 'profiling_results.json')
    if os.path.exists(profiling_path):
        ax = fig.add_subplot(gs[2, 2:])

        with open(profiling_path, 'r') as f:
            profiling = json.load(f)

        operations = ['Direct Blending', 'Pyramid (RGB)', 'Pyramid (LAB)', 'Pyramid (YUV)']
        times = [
            profiling.get('direct_blending', {}).get('time_ms', 0),
            profiling.get('pyramid_blending_rgb', {}).get('time_ms', 0),
            profiling.get('pyramid_blending_lab', {}).get('time_ms', 0),
            profiling.get('pyramid_blending_yuv', {}).get('time_ms', 0)
        ]

        bars = ax.barh(operations, times, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        ax.set_xlabel('Processing Time (ms)', fontweight='bold')
        ax.set_title('Processing Speed Comparison', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f} ms',
                   ha='left', va='center', fontsize=9, fontweight='bold')

    # --- Row 4: Summary table ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Load additional statistics
    boundary_path = os.path.join(output_dir, 'boundary_analysis', 'boundary_statistics.json')
    artifact_path = os.path.join(output_dir, 'artifact_analysis', 'artifact_statistics.json')

    boundary_stats = {}
    artifact_stats = {}

    if os.path.exists(boundary_path):
        with open(boundary_path, 'r') as f:
            boundary_stats = json.load(f)

    if os.path.exists(artifact_path):
        with open(artifact_path, 'r') as f:
            artifact_stats = json.load(f)

    # Create summary table
    table_data = [['Method', 'SSIM ↑', 'MSE ↓', 'PSNR (dB) ↑', 'Boundary Grad ↓', 'Blockiness ↓', 'Overall Rank']]

    for key, name in methods.items():
        metric_key = key.replace('pyramid_', '')  # pyramid_to_L0 -> to_L0
        if key == 'direct':
            m = metrics.get('direct_blending', {})
        elif metric_key in metrics:
            m = metrics[metric_key]
        elif key in metrics:
            m = metrics[key]
        else:
            continue

        ssim = m.get('ssim', 0)
        mse = m.get('mse', 0)
        psnr = m.get('psnr', 0)

        # Boundary stats (old format used 0level, 5level)
        # Convert to_L0 -> 0level for compatibility
        if 'to_L' in metric_key:
            old_format_key = f'pyramid_{metric_key.replace("to_L", "") + "level"}'
        else:
            old_format_key = key if key == 'direct' else f'pyramid_{metric_key}'
        boundary_grad = boundary_stats.get(old_format_key, {}).get('std', 0)

        # Artifact stats (old format)
        artifact_key = old_format_key
        blockiness = artifact_stats.get(artifact_key, {}).get('blockiness', 0)

        # Calculate overall rank (lower is better, normalize each metric)
        # SSIM: higher is better, so use (1-ssim) for ranking
        # Others: lower is better
        rank_score = (1-ssim) + mse*100 + boundary_grad*10 + blockiness*10

        table_data.append([
            name,
            f"{ssim:.4f}" if isinstance(ssim, (int, float)) and ssim else "N/A",
            f"{mse:.5f}" if isinstance(mse, (int, float)) and mse else "N/A",
            f"{psnr:.2f}" if isinstance(psnr, (int, float)) and psnr and psnr < 999 else "N/A",
            f"{boundary_grad:.5f}" if isinstance(boundary_grad, (int, float)) and boundary_grad else "N/A",
            f"{blockiness:.5f}" if isinstance(blockiness, (int, float)) and blockiness else "N/A",
            f"{rank_score:.3f}" if isinstance(rank_score, (int, float)) else "N/A"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white')

    plt.suptitle('Pyramid Blending: Comprehensive Method Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Save
    output_path = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Comprehensive comparison saved: {output_path}")

    print("\n" + "="*80)
    print(" "*15 + "✓ COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*80)
    print()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    create_comprehensive_comparison(output_dir)
