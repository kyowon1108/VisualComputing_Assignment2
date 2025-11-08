"""
ROI Heatmap Analysis 실행 스크립트
"""
import os
from src.preprocessing import load_and_preprocess, create_mask
from src.utils import load_image
from src.heatmap_analysis import create_all_heatmaps


def main():
    """Run heatmap analysis on all blending results"""
    print("\n" + "="*80)
    print(" "*20 + "ROI HEATMAP ANALYSIS")
    print("="*80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    blending_dir = os.path.join(output_dir, 'blending_results')

    # 1. Load preprocessed images
    print("\n[Step 1] Loading images...")
    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')
    hand_img, eye_img = load_and_preprocess(hand_path, eye_path)
    print(f"  ✓ Hand: {hand_img.shape}")
    print(f"  ✓ Eye: {eye_img.shape}")

    # 2. Create mask
    print("\n[Step 2] Creating mask...")
    mask = create_mask(shape=(480, 640),
                      center=(325, 315),
                      axes=(48, 36),
                      blur_kernel=31,
                      output_dir=None)
    print(f"  ✓ Mask: {mask.shape}")

    # 3. Load blending results
    print("\n[Step 3] Loading blending results...")
    methods_dict = {}

    # Direct blending (reference)
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

    if 'direct' not in methods_dict:
        print("\n❌ Direct blending result not found!")
        print("Please run main.py first to generate blending results.")
        return

    # 4. Create heatmaps
    print("\n[Step 4] Creating heatmaps...")
    reference = methods_dict['direct']
    create_all_heatmaps(methods_dict, reference, mask, output_dir)

    print("\n" + "="*80)
    print(" "*20 + "✓ HEATMAP ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {os.path.join(output_dir, 'heatmap_analysis')}/")
    print()


if __name__ == '__main__':
    main()
