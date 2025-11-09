"""
Generate ROI comparison images for all pyramid levels
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import load_image
from src.preprocessing import create_mask

def create_all_levels_roi_comparison():
    """Generate ROI comparison image for all levels"""
    
    # ROI definitions
    ROI_HAND = {
        'center': (400, 130),
        'size': (60, 100),
        'name': 'ROI-1: Hand Region'
    }
    
    ROI_EYE = {
        'center': (315, 325),
        'size': (115, 90),
        'name': 'ROI-2: Eye Region'
    }
    
    ROI_BOUNDARY = {
        'center': (315, 325),
        'outer_radius': (119, 89),
        'inner_radius': (90, 67),
        'name': 'ROI-3: Boundary Ring'
    }
    
    # Load images
    methods = ['direct', 'pyramid_to_L0', 'pyramid_to_L1', 'pyramid_to_L2', 
               'pyramid_to_L3', 'pyramid_to_L4', 'pyramid_to_L5']
    method_names = ['Direct', 'Pyramid L0\n(Full)', 'Pyramid L1', 'Pyramid L2',
                    'Pyramid L3', 'Pyramid L4', 'Pyramid L5\n(Coarse)']
    
    images = {}
    for method in methods:
        if method == 'direct':
            path = 'output/blending_results/direct_blend.jpg'
        else:
            level = method.replace('pyramid_to_L', '')
            path = f'output/blending_results/pyramid_blend_to_L{level}.jpg'
        
        if os.path.exists(path):
            images[method] = load_image(path)
            print(f'  ✓ Loaded {method}')
    
    # Create figure: 3 ROIs × 7 methods = 21 subplots
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(3, 7, figure=fig, hspace=0.3, wspace=0.2)
    
    # ROI extraction function
    def extract_roi(img, roi_def):
        if 'size' in roi_def:
            # Rectangle ROI
            cx, cy = roi_def['center']
            w, h = roi_def['size']
            x = cx - w // 2
            y = cy - h // 2
            return img[y:y+h, x:x+w]
        else:
            # Ellipse ROI (Boundary)
            import cv2
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cx, cy = roi_def['center']
            outer_r = roi_def['outer_radius']
            inner_r = roi_def['inner_radius']
            
            # Outer ellipse
            cv2.ellipse(mask, (cx, cy), outer_r, 0, 0, 360, 255, -1)
            # Subtract inner ellipse
            inner_mask = np.zeros_like(mask)
            cv2.ellipse(inner_mask, (cx, cy), inner_r, 0, 0, 360, 255, -1)
            mask[inner_mask == 255] = 0
            
            # Bounding box
            ys, xs = np.where(mask == 255)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            
            # Extract and apply mask
            roi = img[y_min:y_max+1, x_min:x_max+1].copy()
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Set non-ROI pixels to black
            for c in range(3):
                roi[:, :, c][mask_crop == 0] = 0
            
            return roi
    
    # ROI list
    rois = [
        ('Hand', ROI_HAND),
        ('Eye', ROI_EYE),
        ('Boundary', ROI_BOUNDARY)
    ]
    
    # For each ROI
    for roi_idx, (roi_name, roi_def) in enumerate(rois):
        # For each method
        for method_idx, (method, display_name) in enumerate(zip(methods, method_names)):
            if method not in images:
                continue
            
            img = images[method]
            roi_crop = extract_roi(img, roi_def)
            
            ax = fig.add_subplot(gs[roi_idx, method_idx])
            ax.imshow(roi_crop)
            
            # Show ROI name on first column only
            if method_idx == 0:
                ax.set_ylabel(roi_def['name'], fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # Show method name on first row only
            if roi_idx == 0:
                ax.set_title(display_name, fontsize=10, fontweight='bold')
            
            ax.axis('off')
    
    plt.suptitle('ROI Comparison: All Pyramid Levels', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path = 'output/roi_analysis/roi_comparison_all_levels.png'
    os.makedirs('output/roi_analysis', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ ROI comparison (all levels) saved: {output_path}')
    
    return output_path

if __name__ == '__main__':
    print('\n' + '='*80)
    print(' '*20 + 'ROI COMPARISON - ALL LEVELS')
    print('='*80)
    print()
    
    create_all_levels_roi_comparison()
    
    print()
    print('='*80)
    print(' '*20 + '✓ COMPLETE!')
    print('='*80)
    print()
