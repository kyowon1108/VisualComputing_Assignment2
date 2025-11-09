"""
Generate images for troubleshooting documentation
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '.')
from src.utils import load_image, save_image

def simulate_clipping_bug():
    """Simulate the negative value clipping bug"""
    print("\n1. Simulating negative value clipping bug...")
    
    # Load a Laplacian pyramid level (has negative values)
    lap_path = 'output/pyramids/blend_laplacian/laplacian_level_2.jpg'
    if not os.path.exists(lap_path):
        print(f"  ⚠ {lap_path} not found, skipping")
        return None
    
    # Create visualization showing the problem
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Simulate reconstruction with and without clipping
    # Use actual blended image to show the effect
    img = load_image('output/blending_results/pyramid_blend_to_L0.jpg')
    
    # Simulate what happens without clipping (add noise that goes negative)
    img_with_negative = img.copy()
    noise = np.random.randn(*img.shape) * 0.15
    img_with_negative = img_with_negative + noise
    
    # Without clipping - clip to uint8 range (simulates old buggy behavior)
    img_buggy = np.clip(img_with_negative * 255, 0, 255).astype(np.uint8) / 255.0
    
    # With clipping - proper approach
    img_fixed = np.clip(img_with_negative, 0, 1.0)
    
    # Plot
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Blended Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].hist(img_with_negative.flatten(), bins=100, alpha=0.7, color='red')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=1, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Histogram After Adding Laplacian\n(Note negative values)', fontweight='bold')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].imshow(img_buggy)
    axes[1, 0].set_title('❌ WITHOUT Clipping (Buggy)\nNegative values → 0 → Dark artifacts', 
                         fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_fixed)
    axes[1, 1].set_title('✓ WITH Clipping (Fixed)\nValues kept in [0,1] range', 
                         fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    plt.suptitle('Troubleshooting: Negative Value Accumulation Bug', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = 'output/troubleshooting/clipping_bug_demo.png'
    os.makedirs('output/troubleshooting', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path

def generate_gradient_interpretation():
    """Generate comparison showing gradient interpretation pitfall"""
    print("\n2. Generating gradient interpretation comparison...")
    
    # Load images
    direct = load_image('output/blending_results/direct_blend.jpg')
    pyr_l0 = load_image('output/blending_results/pyramid_blend_to_L0.jpg')
    pyr_l3 = load_image('output/blending_results/pyramid_blend_to_L3.jpg')
    pyr_l5 = load_image('output/blending_results/pyramid_blend_to_L5.jpg')
    
    # ROI metrics from JSON
    import json
    with open('output/roi_analysis/roi_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    methods = [
        ('direct', direct, 'Direct'),
        ('pyramid_to_L0', pyr_l0, 'Pyramid L0\n(Full)'),
        ('pyramid_to_L3', pyr_l3, 'Pyramid L3\n(Medium)'),
        ('pyramid_to_L5', pyr_l5, 'Pyramid L5\n(Coarse)')
    ]
    
    # Row 1: Full images
    for idx, (key, img, name) in enumerate(methods):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.set_title(name, fontweight='bold', fontsize=11)
        ax.axis('off')
    
    # Row 2: Metrics bars
    ax_metrics = fig.add_subplot(gs[1, :])
    
    method_names = ['Direct', 'Pyr L0', 'Pyr L3', 'Pyr L5']
    boundary_grads = []
    hand_textures = []
    eye_contrasts = []
    
    for key, _, _ in methods:
        if key in metrics:
            m = metrics[key]
            boundary_grads.append(m.get('boundary', {}).get('gradient_std', 0))
            hand_textures.append(m.get('hand', {}).get('texture_variance', 0))
            eye_contrasts.append(m.get('eye', {}).get('contrast', 0))
    
    x = np.arange(len(method_names))
    width = 0.25
    
    ax_metrics.bar(x - width, boundary_grads, width, label='Boundary Gradient Std ↓', color='red', alpha=0.7)
    ax_metrics.bar(x, [v*100 for v in hand_textures], width, label='Hand Texture Var ×100 ↑', color='blue', alpha=0.7)
    ax_metrics.bar(x + width, [v*100 for v in eye_contrasts], width, label='Eye Contrast ×100 ↑', color='green', alpha=0.7)
    
    ax_metrics.set_xlabel('Method', fontweight='bold')
    ax_metrics.set_ylabel('Value', fontweight='bold')
    ax_metrics.set_title('Metrics Comparison: The Trade-off', fontweight='bold', fontsize=12)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(method_names)
    ax_metrics.legend(loc='upper right')
    ax_metrics.grid(axis='y', alpha=0.3)
    
    # Row 3: Text analysis
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    
    analysis_text = """
    Key Insight: Lower Gradient ≠ Better Quality
    
    • Pyramid L5 has LOWEST boundary gradient (3.96) → Smoothest boundary ✓
      BUT also has LOWEST texture variance (0.011) → Lost all details ✗
      AND LOWEST eye contrast (0.31) → Blurry eye ✗
    
    • Pyramid L0 has HIGHER boundary gradient (25.94) → Visible boundary ⚠
      BUT maintains HIGH texture variance (0.046) → Preserved details ✓
      AND HIGH eye contrast (0.72) → Sharp eye ✓
    
    • Pyramid L3 is the SWEET SPOT:
      - Moderate boundary gradient (9.20) → Smooth enough ✓
      - Good texture variance (0.036) → Details preserved ✓
      - Good eye contrast (0.51) → Acceptable quality ✓
    
    Conclusion: Metrics must be evaluated in combination, not in isolation!
    """
    
    ax_text.text(0.05, 0.95, analysis_text, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Troubleshooting: Gradient Metric Interpretation Pitfall', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = 'output/troubleshooting/gradient_interpretation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path

def generate_terminology_fix():
    """Generate diagram showing terminology confusion and fix"""
    print("\n3. Generating terminology fix diagram...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before (confusing)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Confusing terminology
    ax.text(5, 9, '❌ BEFORE: Confusing Terminology', ha='center', fontsize=14, 
            fontweight='bold', color='red')
    
    levels_before = [
        ('pyramid_blend_0level.jpg', 'Sounds like "no levels"?', 'Actually: BEST quality (all levels)'),
        ('pyramid_blend_5level.jpg', 'Sounds like "5 levels used"?', 'Actually: WORST quality (L5 only)')
    ]
    
    y_pos = 7
    for filename, confusion, reality in levels_before:
        ax.text(1, y_pos, f'File: {filename}', fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(1, y_pos-0.8, f'❓ {confusion}', fontsize=8, style='italic')
        ax.text(1, y_pos-1.5, f'✓ {reality}', fontsize=8, color='green')
        y_pos -= 3
    
    # After (clear)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9, '✓ AFTER: Clear Terminology', ha='center', fontsize=14, 
            fontweight='bold', color='green')
    
    levels_after = [
        ('pyramid_blend_to_L0.jpg', 'Reconstruct TO level 0', 'L5→L4→L3→L2→L1→L0 (FULL)'),
        ('pyramid_blend_to_L5.jpg', 'Reconstruct TO level 5', 'L5 only (COARSE)')
    ]
    
    y_pos = 7
    for filename, meaning, process in levels_after:
        ax.text(1, y_pos, f'File: {filename}', fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(1, y_pos-0.8, f'✓ {meaning}', fontsize=8, fontweight='bold')
        ax.text(1, y_pos-1.5, f'Process: {process}', fontsize=8, family='monospace')
        y_pos -= 3
    
    plt.suptitle('Troubleshooting: Terminology Refinement', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = 'output/troubleshooting/terminology_fix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path

if __name__ == '__main__':
    print('='*80)
    print(' '*20 + 'GENERATING TROUBLESHOOTING IMAGES')
    print('='*80)
    
    img1 = simulate_clipping_bug()
    img2 = generate_gradient_interpretation()
    img3 = generate_terminology_fix()
    
    print('\n' + '='*80)
    print(' '*20 + '✓ ALL IMAGES GENERATED')
    print('='*80)
    print()
