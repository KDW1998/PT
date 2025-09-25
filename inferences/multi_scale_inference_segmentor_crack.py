'''
python inferences/multi_scale_inference_segmentor_crack.py --crack_config  "/home/user/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/CutMix/BR_Joint_leakage/100units/nonoversampling/seg_PS800_HNS1200_CutMix_noOversampling/convnext_tiny_fpn_crack_hardnegative_100units.py" --crack_checkpoint "/home/user/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/CutMix/BR_Joint_leakage/100units/nonoversampling/seg_PS800_HNS1200_CutMix_noOversampling/iter_joint(63000).pth" --srx_dir "/home/user/WindowsShare/05. Data/02. Training&Test/022. Ministry of Ocean/009_2023.0_건설연제공_매핑용/관리번호 #3, #4_카페리부두_해측부" --rst_dir "home/user/PT/Examples" --srx_suffix ".JPG" --rst_suffix ".JPG" --mask_suffix ".png "--rgb_to_bgr
'''

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
from mmengine import track_progress

from quantify_seg_results import quantify_crack_width_length

from torch.cuda import empty_cache
from utils import inference_segmentor_sliding_window

def parse_args():
    parser = argparse.ArgumentParser(description='Inference detector')
    parser.add_argument('--crack_config', help='the config file to inference crack')
    parser.add_argument('--crack_checkpoint', help='the checkpoint file to inference crack')
    parser.add_argument('--srx_dir', help='the dir to inference')
    parser.add_argument('--rst_dir', help='the dir to save result')
    parser.add_argument('--srx_suffix', default='.png', help='the source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='the result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='the mask output extension')
    parser.add_argument('--alpha', default=0.8, help='the alpha value for blending')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='convert rgb to bgr, if the model palette is written in rgb format.')
    parser.set_defaults(rgb_to_bgr=False)
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='overwrite the crack palette with black and red. To be used when the crack model is trained with a different palette.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')

    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))

    crack_palette = crack_model.dataset_meta['palette'][:2]  # Only keep the first two colors

    if args.rgb_to_bgr:
        crack_palette = [p[::-1] for p in crack_palette]

    if args.overwrite_crack_palette:
        crack_palette[1] = [0, 0, 255]  # Redefine crack color if necessary

    for img_path in img_list:
        _, crack_mask = inference_segmentor_sliding_window(crack_model, img_path, color_mask=None, score_thr=0.1, window_size=1024, overlap_ratio=0.1)

        seg_result = mmcv.imread(img_path)

        # Visualize the crack mask
        color = crack_palette[1]
        color = np.array(color, dtype=np.uint8)
        mask_bool = crack_mask == 1

        seg_result[mask_bool, :] = seg_result[mask_bool, :] * (1 - args.alpha) + color * args.alpha

        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        # Quantify crack width and length
        seg_result = quantify_crack_width_length(seg_result, crack_mask, crack_palette[1])

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(crack_mask.astype(np.uint8), mask_path)  # Assuming binary mask for simplicity

if __name__ == '__main__':
    main()
