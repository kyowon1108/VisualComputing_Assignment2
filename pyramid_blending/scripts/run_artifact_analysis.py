"""
Artifact Analysis 실행 스크립트
"""
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from src.preprocessing import create_mask
from src.utils import load_image
from src.artifact_analysis import create_artifact_analysis_report


def main():
    """Run artifact analysis"""
    print("\n" + "="*80)
    print(" "*20 + "ARTIFACT ANALYSIS")
    print("="*80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    blending_dir = os.path.join(output_dir, 'blending_results')

    # Load blending results
    print("\n[Step 1] Loading blending results...")
    methods_dict = {}

    # Direct blending
    direct_path = os.path.join(blending_dir, 'direct_blend.jpg')
    if os.path.exists(direct_path):
        methods_dict['direct'] = load_image(direct_path)
        print(f"  ✓ Direct blending")

    # Pyramid levels
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
        return

    # Analyze artifacts
    print("\n[Step 2] Analyzing artifacts...")
    create_artifact_analysis_report(methods_dict, output_dir)

    print("\n" + "="*80)
    print(" "*20 + "✓ ARTIFACT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {os.path.join(output_dir, 'artifact_analysis')}/")
    print()


if __name__ == '__main__':
    main()
