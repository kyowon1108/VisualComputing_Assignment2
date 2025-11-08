"""
Artifact Analysis Module

블렌딩 아티팩트 (Blockiness, Ghost, Color Fringing) 시각적 예시
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def measure_blockiness(image, block_size=8):
    """
    Measure blockiness artifact using block boundary discontinuity

    Args:
        image: Image (H, W, 3) in [0, 1]
        block_size: Block size for blockiness measurement

    Returns:
        blockiness_score: Higher = more blocky
        blockiness_map: Map of block boundaries
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    h, w = image_gray.shape

    # Calculate horizontal block boundaries
    h_diff = []
    for i in range(block_size, h, block_size):
        if i < h:
            diff = np.abs(image_gray[i, :] - image_gray[i-1, :])
            h_diff.append(np.mean(diff))

    # Calculate vertical block boundaries
    v_diff = []
    for j in range(block_size, w, block_size):
        if j < w:
            diff = np.abs(image_gray[:, j] - image_gray[:, j-1])
            v_diff.append(np.mean(diff))

    # Blockiness score
    blockiness_score = np.mean(h_diff + v_diff) if (h_diff + v_diff) else 0.0

    # Create blockiness map
    blockiness_map = np.zeros_like(image_gray)

    # Highlight block boundaries
    for i in range(block_size, h, block_size):
        if i < h:
            blockiness_map[i, :] = np.abs(image_gray[i, :] - image_gray[i-1, :])

    for j in range(block_size, w, block_size):
        if j < w:
            blockiness_map[:, j] = np.maximum(
                blockiness_map[:, j],
                np.abs(image_gray[:, j] - image_gray[:, j-1])
            )

    return blockiness_score, blockiness_map


def detect_ghost_artifact(image1, image2, threshold=0.05):
    """
    Detect ghost artifacts (double edges)

    Args:
        image1: First image (H, W, 3)
        image2: Second image (H, W, 3)
        threshold: Difference threshold

    Returns:
        ghost_map: Map showing potential ghost artifacts
    """
    # Calculate absolute difference
    diff = np.abs(image1 - image2)

    if len(diff.shape) == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff

    # Detect edges in difference
    edges = cv2.Canny((diff_gray * 255).astype(np.uint8), 50, 150)

    # Ghost map (edges in difference indicate ghosting)
    ghost_map = edges.astype(np.float32) / 255.0

    return ghost_map


def visualize_artifacts(methods_dict, output_path):
    """
    Create comprehensive artifact visualization

    Args:
        methods_dict: Dictionary of method_name -> image
        output_path: Path to save visualization
    """
    # Analyze artifacts for each method
    artifact_stats = {}

    for method_name, image in methods_dict.items():
        blockiness, _ = measure_blockiness(image)
        artifact_stats[method_name] = {
            'blockiness': float(blockiness)  # Convert to native Python float
        }

    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # --- Row 1: Cropped regions showing artifacts ---
    methods_to_show = ['direct', 'pyramid_0level', 'pyramid_3level', 'pyramid_5level']

    # Define crop regions for artifact inspection
    crop_regions = [
        ('Eye Detail', (280, 120, 380, 200)),  # Eye region
        ('Boundary', (300, 200, 400, 280)),    # Boundary region
        ('Hand', (380, 90, 460, 170))          # Hand region
    ]

    for row_idx, (region_name, (x1, y1, x2, y2)) in enumerate(crop_regions):
        for col_idx, method in enumerate(methods_to_show):
            if method not in methods_dict:
                continue

            ax = fig.add_subplot(gs[row_idx, col_idx])
            image = methods_dict[method]

            # Crop region
            crop = image[y1:y2, x1:x2]

            ax.imshow(crop)
            title = f'{method.replace("_", " ").title()}'
            if col_idx == 0:
                title = f'{region_name}\n{title}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

            # Add zoom indicator on first method
            if col_idx == 0 and row_idx == 0:
                ax.text(0.02, 0.98, f'Location: ({x1}, {y1})',
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # --- Row 4: Blockiness comparison ---
    ax = fig.add_subplot(gs[3, :2])

    method_names = list(artifact_stats.keys())
    blockiness_scores = [artifact_stats[m]['blockiness'] for m in method_names]

    x_pos = np.arange(len(method_names))
    bars = ax.bar(x_pos, blockiness_scores, color='coral', alpha=0.8)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Blockiness Score', fontweight='bold')
    ax.set_title('Blockiness Artifact Measurement\n(Lower = Better)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', '\n') for m in method_names],
                       rotation=0, ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.6f}',
               ha='center', va='bottom', fontsize=8)

    # Highlight best (lowest blockiness)
    best_idx = blockiness_scores.index(min(blockiness_scores))
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)

    # --- Row 4 Col 3-4: Artifact description ---
    ax = fig.add_subplot(gs[3, 2:])
    ax.axis('off')

    artifact_info = """
    ARTIFACT TYPES & OBSERVATIONS:

    1. BLOCKINESS
       - Definition: Block boundary discontinuities
       - Cause: Quantization in reconstruction
       - Observation: Pyramid blending reduces blockiness
       - Measurement: Block boundary gradient

    2. GHOSTING
       - Definition: Double edges or halos
       - Cause: Misalignment or poor mask
       - Observation: More visible in direct blending
       - Detection: Edge detection on difference

    3. COLOR FRINGING
       - Definition: Color artifacts at boundaries
       - Cause: Channel-wise processing differences
       - Solution: LAB color space processing
       - Mitigation: Multi-scale blending

    QUALITY RANKING (Best to Worst):
    """

    # Add ranking
    sorted_methods = sorted(artifact_stats.items(), key=lambda x: x[1]['blockiness'])
    for rank, (method, stats) in enumerate(sorted_methods[:5], 1):
        artifact_info += f"\n    #{rank} {method:<20} (Blockiness: {stats['blockiness']:.6f})"

    ax.text(0.05, 0.95, artifact_info, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Artifact Analysis: Visual Quality Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Artifact visualization saved: {output_path}")

    return artifact_stats


def create_artifact_analysis_report(methods_dict, output_dir):
    """
    Create comprehensive artifact analysis report

    Args:
        methods_dict: Dictionary of method_name -> image
        output_dir: Output directory
    """
    import os
    import json

    artifact_dir = os.path.join(output_dir, 'artifact_analysis')
    os.makedirs(artifact_dir, exist_ok=True)

    print("\n[Artifact Analysis] Analyzing visual artifacts...")

    # Create visualization
    viz_path = os.path.join(artifact_dir, 'artifact_comparison.png')
    artifact_stats = visualize_artifacts(methods_dict, viz_path)

    # Save statistics
    stats_path = os.path.join(artifact_dir, 'artifact_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(artifact_stats, f, indent=2)
    print(f"  ✓ Statistics saved: {stats_path}")

    # Print summary
    print("\n  Artifact Analysis Summary:")
    print("  " + "-"*60)
    sorted_methods = sorted(artifact_stats.items(), key=lambda x: x[1]['blockiness'])
    for rank, (method, stats) in enumerate(sorted_methods, 1):
        print(f"  #{rank} {method:<20} Blockiness: {stats['blockiness']:.6f}")
    print("  " + "-"*60)

    print(f"\n✓ Artifact analysis saved to: {artifact_dir}/")
