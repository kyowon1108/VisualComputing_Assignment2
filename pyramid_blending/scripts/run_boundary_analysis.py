"""
Boundary Quality Analysis 실행 스크립트
"""
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from src.preprocessing import load_and_preprocess, create_mask
from src.utils import load_image
from src.boundary_analysis import create_boundary_analysis_report


def main():
    """Run boundary quality analysis"""
    print("\n" + "="*80)
    print(" "*20 + "BOUNDARY QUALITY ANALYSIS")
    print("="*80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    blending_dir = os.path.join(output_dir, 'blending_results')

    # 1. Create mask
    print("\n[Step 1] Creating mask...")
    mask = create_mask(shape=(480, 640),
                      center=(325, 315),
                      axes=(48, 36),
                      blur_kernel=31,
                      output_dir=None)
    print(f"  ✓ Mask: {mask.shape}")

    # 2. Load blending results
    print("\n[Step 2] Loading blending results...")
    methods_dict = {}

    # Direct blending
    direct_path = os.path.join(blending_dir, 'direct_blend.jpg')
    if os.path.exists(direct_path):
        methods_dict['direct'] = load_image(direct_path)
        print(f"  ✓ Direct blending")

    # Pyramid levels (0-5)
    for level in range(6):
        pyramid_path = os.path.join(blending_dir, f'pyramid_blend_{level}level.jpg')
        if os.path.exists(pyramid_path):
            methods_dict[f'pyramid_{level}level'] = load_image(pyramid_path)
            print(f"  ✓ Pyramid {level}-level")

    # LAB blending
    lab_path = os.path.join(blending_dir, 'lab_blend_5level.jpg')
    if os.path.exists(lab_path):
        methods_dict['lab_5level'] = load_image(lab_path)
        print(f"  ✓ LAB 5-level")

    print(f"\n  Total {len(methods_dict)} methods loaded")

    if len(methods_dict) == 0:
        print("\n❌ No blending results found!")
        print("Please run main.py first.")
        return

    # 3. Analyze boundary quality
    print("\n[Step 3] Analyzing boundary quality...")
    create_boundary_analysis_report(methods_dict, mask, output_dir)

    print("\n" + "="*80)
    print(" "*20 + "✓ BOUNDARY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {os.path.join(output_dir, 'boundary_analysis')}/")
    print()


if __name__ == '__main__':
    main()
