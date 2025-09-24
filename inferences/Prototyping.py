'''
Crack Detection and Quantification with CSV Export
폴더 내 모든 이미지를 탐지하고 정량화 결과를 CSV로 저장하는 코드
'''
import os
import csv
import argparse
from glob import glob
from pathlib import Path

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
from mmengine import track_progress

from quantify_seg_results import quantify_crack_width_length
from torch.cuda import empty_cache
from utils import inference_segmentor_sliding_window

def parse_args():
    parser = argparse.ArgumentParser(description='Crack Detection and Quantification with CSV Export')
    parser.add_argument('--crack_config', help='the config file to inference crack')
    parser.add_argument('--crack_checkpoint', help='the checkpoint file to inference crack')
    parser.add_argument('--srx_dir', help='the dir to inference')
    parser.add_argument('--rst_dir', help='the dir to save result')
    parser.add_argument('--csv_output', default='crack_detection_results.csv', help='the CSV output file name')
    parser.add_argument('--srx_suffix', default='.png', help='the source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='the result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='the mask output extension')
    parser.add_argument('--alpha', default=0.8, help='the alpha value for blending')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='convert rgb to bgr, if the model palette is written in rgb format.')
    parser.set_defaults(rgb_to_bgr=False)
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='overwrite the crack palette with black and red. To be used when the crack model is trained with a different palette.')
    parser.add_argument('--minimum_area', default=500, help='minimum crack area for detection')

    args = parser.parse_args()
    return args

def process_single_image(crack_model, img_path, args, crack_palette):
    """
    Process a single image for crack detection and quantification
    
    Args:
        crack_model: The crack detection model
        img_path: Path to the input image
        args: Command line arguments
        crack_palette: Color palette for crack visualization
        
    Returns:
        tuple: (image_name, crack_quantification_results)
    """
    try:
        # Perform crack detection
        _, crack_mask = inference_segmentor_sliding_window(
            crack_model, img_path, color_mask=None, 
            score_thr=0.1, window_size=1024, overlap_ratio=0.1
        )

        # Read the original image
        seg_result = mmcv.imread(img_path)
        
        # Visualize the crack mask
        color = crack_palette[1]
        color = np.array(color, dtype=np.uint8)
        mask_bool = crack_mask == 1

        seg_result[mask_bool, :] = seg_result[mask_bool, :] * (1 - args.alpha) + color * args.alpha

        # Quantify crack width and length
        seg_result, crack_quantification_results = quantify_crack_width_length(
            seg_result, crack_mask, crack_palette[1], 
            minimum_area=args.minimum_area
        )

        # Save result images
        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        # Create output directory if it doesn't exist
        os.makedirs(args.rst_dir, exist_ok=True)
        
        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(crack_mask.astype(np.uint8), mask_path)

        # Get image name without extension
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        return image_name, crack_quantification_results

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, []

def save_results_to_csv(all_results, csv_output_path):
    """
    Save all crack detection results to CSV file
    
    Args:
        all_results: List of tuples containing (image_name, crack_quantification_results)
        csv_output_path: Path to save the CSV file
    """
    # Prepare CSV data - only include images with detected cracks
    csv_data = []
    
    for image_name, crack_results in all_results:
        if crack_results:  # Only save if cracks were detected
            for crack_data in crack_results:
                csv_data.append({
                    'Image_Name': image_name,
                    'Pixel_Coordinates': crack_data[0],
                    'Measurements': crack_data[1],
                    'Class_ID': crack_data[2]
                })
    
    # Write to CSV
    fieldnames = ['Image_Name', 'Pixel_Coordinates', 'Measurements', 'Class_ID']
    
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Results saved to: {csv_output_path}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with cracks detected: {len(csv_data)}")
    print(f"Total crack detections: {sum(len(results) for _, results in all_results if results)}")

def main():
    args = parse_args()
    
    # Initialize crack detection model
    print("Initializing crack detection model...")
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')
    
    # Get list of images to process
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))
    
    if not img_list:
        print(f"No images found in {args.srx_dir} with suffix {args.srx_suffix}")
        return
    
    print(f"Found {len(img_list)} images to process")
    
    # Setup crack palette
    crack_palette = crack_model.dataset_meta['palette'][:2]  # Only keep the first two colors

    if args.rgb_to_bgr:
        crack_palette = [p[::-1] for p in crack_palette]

    if args.overwrite_crack_palette:
        crack_palette[1] = [0, 0, 255]  # Redefine crack color if necessary

    # Process all images
    all_results = []
    
    print("Processing images...")
    for i, img_path in enumerate(img_list):
        print(f"Processing {i+1}/{len(img_list)}: {os.path.basename(img_path)}")
        
        image_name, crack_results = process_single_image(crack_model, img_path, args, crack_palette)
        
        if image_name is not None:
            all_results.append((image_name, crack_results))
        
        # Clear GPU memory
        empty_cache()
    
    # Save results to CSV
    csv_output_path = os.path.join(args.rst_dir, args.csv_output)
    save_results_to_csv(all_results, csv_output_path)
    
    print("Processing completed!")

if __name__ == '__main__':
    main()
