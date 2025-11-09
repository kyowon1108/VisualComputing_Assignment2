"""
ROI Analysis 실행 스크립트
기존 블렌딩 결과들에 대해 ROI 분석 수행
"""
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
from src.roi_analysis import analyze_all_methods, create_roi_summary_table
from src.preprocessing import load_and_preprocess, create_mask
from src.utils import load_image


def main():
    """ROI 분석 실행"""
    print("\n" + "="*80)
    print(" "*20 + "ROI ANALYSIS FOR PYRAMID BLENDING")
    print("="*80)

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    blending_dir = os.path.join(output_dir, 'blending_results')

    # 1. Load preprocessed images
    print("\n[Step 1] 이미지 로딩...")
    hand_path = os.path.join(base_dir, 'input', 'hand_raw.jpg')
    eye_path = os.path.join(base_dir, 'input', 'eye_raw.jpg')

    hand_img, eye_img = load_and_preprocess(hand_path, eye_path)
    print(f"  ✓ Hand: {hand_img.shape}")
    print(f"  ✓ Eye: {eye_img.shape}")

    # 2. Create mask
    print("\n[Step 2] 마스크 생성...")
    mask = create_mask(shape=(480, 640),
                      center=(325, 315),
                      axes=(48, 36),
                      blur_kernel=31,
                      output_dir=None)
    print(f"  ✓ Mask: {mask.shape}")

    # 3. Load blending results
    print("\n[Step 3] 블렌딩 결과 로딩...")
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

    print(f"\n  총 {len(methods_dict)}개 방법 로딩 완료")

    if len(methods_dict) == 0:
        print("\n❌ 블렌딩 결과를 찾을 수 없습니다!")
        print("먼저 main.py를 실행하여 블렌딩 결과를 생성하세요.")
        return

    # 4. Analyze ROI for all methods
    print("\n[Step 4] ROI 분석 시작...")
    all_results = analyze_all_methods(methods_dict, mask, output_dir, hand_img, eye_img)

    # 5. Create summary table
    print("\n[Step 5] 요약 테이블 생성...")
    roi_dir = os.path.join(output_dir, 'roi_analysis')
    table_path = os.path.join(roi_dir, 'roi_summary_table.png')
    create_roi_summary_table(all_results, table_path)

    # 6. Print key findings
    print("\n" + "="*80)
    print(" "*25 + "KEY FINDINGS")
    print("="*80)

    # Find best method for each ROI
    best_hand_method = None
    best_hand_ssim = -1

    best_boundary_method = None
    best_boundary_grad = float('inf')

    for method_name, metrics in all_results.items():
        # Hand region
        if metrics.get('hand') and metrics['hand'].get('ssim'):
            ssim_val = metrics['hand']['ssim']
            if ssim_val > best_hand_ssim:
                best_hand_ssim = ssim_val
                best_hand_method = method_name

        # Boundary region
        if metrics.get('boundary') and metrics['boundary'].get('gradient_std'):
            grad_std = metrics['boundary']['gradient_std']
            if grad_std < best_boundary_grad:
                best_boundary_grad = grad_std
                best_boundary_method = method_name

    print("\n🏆 Best Method per ROI:")
    print("-"*80)
    if best_hand_method:
        print(f"  ROI-1 (Hand):     {best_hand_method:<20} (SSIM: {best_hand_ssim:.4f})")
    if best_boundary_method:
        status = "✓✓✓ Excellent" if best_boundary_grad < 10.0 else "✓ Good"
        print(f"  ROI-3 (Boundary): {best_boundary_method:<20} (Grad Std: {best_boundary_grad:.4f} {status})")

    # Compare Direct vs Pyramid 0-level
    if 'direct' in all_results and 'pyramid_0level' in all_results:
        print("\n📊 Direct vs Pyramid (Level 0) Comparison:")
        print("-"*80)

        # Boundary region comparison
        if (all_results['direct'].get('boundary') and
            all_results['pyramid_0level'].get('boundary')):

            direct_grad = all_results['direct']['boundary'].get('gradient_std', 'N/A')
            pyramid_grad = all_results['pyramid_0level']['boundary'].get('gradient_std', 'N/A')

            if isinstance(direct_grad, float) and isinstance(pyramid_grad, float):
                improvement = (direct_grad - pyramid_grad) / direct_grad * 100
                print(f"  Boundary Gradient Std:")
                print(f"    Direct:          {direct_grad:.4f}")
                print(f"    Pyramid (L0):    {pyramid_grad:.4f}")
                print(f"    Improvement:     {improvement:.1f}% {'✓✓✓' if improvement > 50 else '✓' if improvement > 0 else ''}")

    print("\n" + "="*80)
    print(" "*20 + "✓ ROI ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {roi_dir}/")
    print("   - roi_metrics.json         : 정량적 메트릭 (JSON)")
    print("   - roi_locations.png        : ROI 위치 시각화 (빨간 박스)")
    print("   - roi_comparison.png       : ROI 확대 비교")
    print("   - roi_report.txt           : 텍스트 리포트")
    print("   - roi_summary_table.png    : 요약 테이블")
    print("\n")


if __name__ == '__main__':
    main()
