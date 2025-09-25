#!/usr/bin/env python3
"""
Overlay and Save Ground Truth Visualization
마스크 시각화를 위한 오버레이 및 저장 스크립트

This script creates translucent red overlays on original images based on crack detection coordinates.
크랙 탐지 좌표를 기반으로 원본 이미지에 반투명 빨간색 오버레이를 생성합니다.

Usage:
    python overlay_and_save_gt.py --img_dir "original_images" --gt_dir "detection_results" --output_dir "visualization_output"
"""

import argparse
import os
import numpy as np
from glob import glob
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Extended color mapping
color_mapping = {
    0: [0, 0, 0],        # Class 0 in black
    1: [0, 0, 255],      # Class 1 in Red (BGR format)
    2: [0, 255, 255],    # Class 2 in Yellow
    3: [255, 0, 0],      # Class 3 in Blue
    4: [0, 255, 0],      # Class 4 in Green
    5: [255, 0, 255],    # Class 5 in Magenta
    6: [255, 255, 0],    # Class 6 in Cyan
    7: [128, 0, 0],      # Class 7 in Maroon
    8: [0, 128, 0]       # Class 8 in Dark Green
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize images and ground truths')
    parser.add_argument('--img_dir', required=True, help='the directory containing original images')
    parser.add_argument('--gt_dir', required=True, help='the directory containing detection result files (coordinates/masks)')
    parser.add_argument('--output_dir', required=True, help='the directory to save the visualized ground truth files')
    parser.add_argument('--target_label', type=int, default=1, help='the label to filter images (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.6, help='transparency for overlay (default: 0.6)')
    parser.add_argument('--img_suffix', default='.JPG', help='original image file suffix')
    parser.add_argument('--gt_suffix', default='.png', help='ground truth/detection file suffix')
    args = parser.parse_args()
    return args

def get_matching_files(img_dir, gt_dir, img_suffix, gt_suffix):
    """Get matching pairs of original images and detection results"""
    img_files = glob(f'{img_dir}/*{img_suffix}')
    gt_files = glob(f'{gt_dir}/*{gt_suffix}')
    
    # Create dictionaries for matching files
    img_dict = {}
    gt_dict = {}
    
    for img_file in img_files:
        base_name = os.path.basename(img_file).replace(img_suffix, '')
        img_dict[base_name] = img_file
    
    for gt_file in gt_files:
        base_name = os.path.basename(gt_file).replace(gt_suffix, '')
        gt_dict[base_name] = gt_file
    
    # Find common base names
    common_keys = set(img_dict.keys()) & set(gt_dict.keys())
    
    matched_pairs = [(img_dict[k], gt_dict[k]) for k in common_keys]
    return matched_pairs

def convert_coordinates_to_mask(coordinates_str, img_shape):
    """
    Convert coordinate string to binary mask
    
    Args:
        coordinates_str: String in format "(minr,minc)-(maxr,maxc)" or similar
        img_shape: Shape of the image (height, width)
    
    Returns:
        binary_mask: Binary mask with detected areas marked as 1
    """
    if not coordinates_str or coordinates_str == "No cracks detected":
        return np.zeros(img_shape[:2], dtype=np.uint8)
    
    try:
        # Parse coordinate string like "(minr,minc)-(maxr,maxc)"
        coords = coordinates_str.replace('(', '').replace(')', '').split('-')
        if len(coords) == 2:
            min_coords = tuple(map(int, coords[0].split(',')))
            max_coords = tuple(map(int, coords[1].split(',')))
            
            # Create binary mask
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            mask[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]] = 1
            return mask
    except:
        pass
    
    return np.zeros(img_shape[:2], dtype=np.uint8)

def create_visualization_overlay(img, gt_mask, color, alpha):
    """
    Create translucent overlay on original image
    
    Args:
        img: Original image (BGR format)
        gt_mask: Binary mask (0s and 1s)
        color: Overlay color (BGR format)
        alpha: Transparency (0.0-1.0)
    
    Returns:
        combined_img: Image with overlay
    """
    # Create color overlay
    gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    gt_rgb[gt_mask == 1] = color
    
    # Create mask for overlay
    mask = np.sum(gt_rgb, axis=2) > 0
    
    # Apply overlay
    combined_img = img.copy()
    combined_img[mask] = cv2.addWeighted(combined_img[mask], 1-alpha, gt_rgb[mask], alpha, 0)
    
    return combined_img

def main():
    args = parse_args()
    img_dir, gt_dir, output_dir, target_label, alpha, img_suffix, gt_suffix = parse_args()
    
    file_pairs = get_matching_files(img_dir, gt_dir, img_suffix, gt_suffix)
    print(f"Found {len(file_pairs)} matching pairs of image and detection files")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for img_file, gt_file in file_pairs:
        try:
            # Read original image
            img = cv2.imread(img_file)
            if img is None:
                print(f"Failed to read image file: {img_file}")
                skipped_count += 1
                continue

            # Read ground truth/detection result
            gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            if gt is None:
                print(f"Failed to read detection file: {gt_file}")
                skipped_count += 1
                continue

            # Check if the target label is present in the ground truth
            if target_label not in np.unique(gt):
                print(f"Skipping {gt_file} - target label {target_label} not found")
                skipped_count += 1
                continue

            # Ensure dimensions match
            if img.shape[:2] != gt.shape[:2]:
                print(f"Resizing detection mask to match image dimensions")
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create binary mask for target label
            gt_mask = (gt == target_label).astype(np.uint8)

            # Create visualization overlay
            color = color_mapping[target_label]
            combined_img = create_visualization_overlay(img, gt_mask, color, alpha)

            # Create side-by-side comparison
            gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            gt_rgb[gt == target_label] = color_mapping[target_label]
            
            combined_side_by_side = np.hstack((combined_img, gt_rgb))

            # Save result
            output_filename = os.path.join(output_dir, os.path.basename(img_file).replace(img_suffix, f'_visualized{img_suffix}'))
            cv2.imwrite(output_filename, combined_side_by_side)
            
            print(f"Processed: {img_file}")
            processed_count += 1

        except Exception as e:
            print(f"Error processing {img_file} and {gt_file}: {str(e)}")
            skipped_count += 1

    print(f"All visualized images were saved to {output_dir}")
    print(f"Processed {processed_count} images")
    print(f"Skipped {skipped_count} images")

if __name__ == '__main__':
    main()
