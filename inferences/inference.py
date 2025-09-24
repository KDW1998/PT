'''
python prototyping/inference.py --crack_config '/home/deogwonkang/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/Only_Positive/seg_PSOnly_Positive800_noOversampling/convnext_tiny_fpn_crack_hardnegative_100units.py' --crack_checkpoint '/home/deogwonkang/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/Only_Positive/seg_PSOnly_Positive800_noOversampling/iter_best.pth' --srx_dir "prototyping/test_data/leftImg8bit/test" --rst_dir "prototyping/test_data/inference_results"
'''
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import mmcv
import numpy as np
import pandas as pd
from mmengine import track_iter_progress
from mmseg.apis import init_model

from quantify_seg_results import quantify_crack_width_length
from utils import inference_segmentor_sliding_window


"""
This script is for multi-scale inference in mmsegmentation, specifically for crack detection.
To enhance the accuracy of crack detection, this script performs inference on high-resolution images.
The crack model should have 'background' and 'crack' classes.
"""

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Inference segmentor for cracks')
    parser.add_argument('--crack_config', required=True, help='config file for crack inference')
    parser.add_argument('--crack_checkpoint', required=True, help='checkpoint file for crack inference')
    parser.add_argument('--srx_dir', required=True, help='directory of images to inference')
    parser.add_argument('--rst_dir', required=True, help='directory to save results')
    parser.add_argument('--srx_suffix', default='.png', help='source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='mask output extension')
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha value for blending')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='convert rgb to bgr if model palette is in rgb format')
    parser.set_defaults(rgb_to_bgr=False)
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='overwrite the crack palette with red')
    parser.add_argument('--scaling_factor', type=float, default=1, help='scaling factor if using upscaled images')
    args = parser.parse_args()
    return args


def main():
    """Main function to run crack inference and quantification."""
    args = parse_args()
    
    # Ensure the result directory exists
    os.makedirs(args.rst_dir, exist_ok=True)

    # Initialize the crack segmentation model
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')

    # Find all images to be processed
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))
    
    # Get palette from the model's metadata
    palette = crack_model.dataset_meta['palette']
    
    if args.rgb_to_bgr:
        palette = [p[::-1] for p in palette]

    if args.overwrite_crack_palette and len(palette) > 1:
        palette[1] = [0, 0, 255] # Set crack color to red (BGR)

    # Initialize list to store all quantification results
    all_quantification_results = []

    # Process each image
    for img_path in track_iter_progress(img_list):
        
        # Perform inference to get the crack mask
        _, crack_mask = inference_segmentor_sliding_window(
            crack_model, img_path, color_mask=None, score_thr=0.1, window_size=2048, overlap_ratio=0.1
        )

        # The final mask is just the crack mask
        mask_result = crack_mask

        # Load the original image to draw the segmentation on
        seg_result = mmcv.imread(img_path)

        # Visualize the crack mask by blending it onto the original image
        if len(palette) > 1:
            color = np.array(palette[1], dtype=np.uint8)
            mask_bool = (mask_result == 1)
            # Apply color blending only on crack pixels
            seg_result[mask_bool, :] = seg_result[mask_bool, :] * (1 - args.alpha) + color * args.alpha

        # Quantify crack properties (e.g., width, length) and draw them on the result image
        seg_result, crack_quantification_results = quantify_crack_width_length(
            seg_result, crack_mask, palette[1]
        )

        # Add image name to each crack result and append to all results
        if crack_quantification_results:
            for result in crack_quantification_results:
                result_with_image = [os.path.basename(img_path)] + result
                all_quantification_results.append(result_with_image)
        
        # Define output paths
        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)
        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        # Save the final visualized image and the raw mask
        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(mask_result, mask_path)

    # Save all quantification results to a single CSV file
    if all_quantification_results:
        all_df = pd.DataFrame(all_quantification_results, columns=['Image_Name', 'Pixel_Coordinates', 'Measurements', 'class_id'])
        csv_path = os.path.join(args.rst_dir, 'all_crack_quantification_results.csv')
        all_df.to_csv(csv_path, index=False)
        print(f"All quantification results saved to: {csv_path}")


if __name__ == '__main__':
    main()