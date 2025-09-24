'''
Enhanced Crack Detection with Size Filtering and Excel Export
개선된 크랙 탐지 시스템 - 크기 필터링 및 Excel 내보내기 기능 포함

사용법:
python enhanced_crack_inference.py --crack_config "config_path" --crack_checkpoint "checkpoint_path" --srx_dir "input_dir" --rst_dir "output_dir"
'''

import os
import argparse
from glob import glob
import pandas as pd
import numpy as np
import mmcv
from mmseg.apis import init_model, inference_model
from mmengine import track_progress
from torch.cuda import empty_cache

# 기존 모듈 import
from quantify_seg_results import quantify_crack_width_length
from utils import inference_segmentor_sliding_window

# 설정 파일 import
from config import CONFIG

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='Enhanced Crack Detection with Excel Export')
    parser.add_argument('--crack_config', required=True, help='크랙 탐지 모델 설정 파일 경로')
    parser.add_argument('--crack_checkpoint', required=True, help='크랙 탐지 모델 체크포인트 파일 경로')
    parser.add_argument('--srx_dir', required=True, help='입력 이미지 디렉토리 경로')
    parser.add_argument('--rst_dir', required=True, help='결과 이미지 저장 디렉토리 경로')
    parser.add_argument('--srx_suffix', default=CONFIG['DEFAULT_INPUT_SUFFIX'], help='입력 이미지 파일 확장자')
    parser.add_argument('--rst_suffix', default=CONFIG['DEFAULT_OUTPUT_SUFFIX'], help='결과 이미지 파일 확장자')
    parser.add_argument('--mask_suffix', default=CONFIG['DEFAULT_MASK_SUFFIX'], help='마스크 이미지 파일 확장자')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='RGB를 BGR로 변환')
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='크랙 팔레트 덮어쓰기')
    
    # Excel 출력 경로 오버라이드 옵션
    parser.add_argument('--excel_output', help='Excel 출력 파일 경로 (기본값: CONFIG에서 설정)')
    
    return parser.parse_args()

def create_output_directories():
    """출력 디렉토리 생성"""
    # 이미지 출력 디렉토리 생성
    os.makedirs(CONFIG['IMAGE_OUTPUT_PATH'], exist_ok=True)
    
    # Excel 출력 디렉토리 생성
    excel_dir = os.path.dirname(CONFIG['EXCEL_OUTPUT_PATH'])
    os.makedirs(excel_dir, exist_ok=True)

def filter_crack_by_size(crack_quantification_results, min_area=None, min_width=None, min_length=None):
    """
    크기 기준으로 크랙 필터링
    
    Args:
        crack_quantification_results: 크랙 정량화 결과 리스트
        min_area: 최소 면적
        min_width: 최소 폭
        min_length: 최소 길이
    
    Returns:
        filtered_results: 필터링된 크랙 결과
    """
    if not crack_quantification_results:
        return []
    
    filtered_results = []
    
    for result in crack_quantification_results:
        # result 형식: [coordinates, measurements, class_id]
        coordinates = result[0]  # "(minr,minc)-(maxr,maxc)" 형식
        measurements = result[1]  # "width x length" 형식
        
        # 측정값 파싱
        try:
            width, length = map(float, measurements.split('x'))
        except:
            continue
        
        # 면적 계산 (대략적)
        coord_parts = coordinates.replace('(', '').replace(')', '').split('-')
        if len(coord_parts) == 2:
            try:
                min_coords = tuple(map(int, coord_parts[0].split(',')))
                max_coords = tuple(map(int, coord_parts[1].split(',')))
                area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
            except:
                area = width * length  # 대안 계산
        else:
            area = width * length
        
        # 크기 필터링
        if min_area and area < min_area:
            continue
        if min_width and width < min_width:
            continue
        if min_length and length < min_length:
            continue
        
        filtered_results.append(result)
    
    return filtered_results

def visualize_crack_detection(seg_result, crack_mask, color=None, alpha=None):
    """
    크랙 탐지 결과를 빨간색 오버레이로 시각화 (정량화 텍스트 제외)
    
    Args:
        seg_result: 원본 이미지
        crack_mask: 크랙 마스크
        color: 오버레이 색상 (BGR 형식)
        alpha: 투명도
    
    Returns:
        visualized_image: 시각화된 이미지
    """
    if color is None:
        color = CONFIG['CRACK_COLOR']
    if alpha is None:
        alpha = CONFIG['VISUALIZATION_ALPHA']
    
    # 크랙 영역에 빨간색 오버레이 적용
    mask_bool = crack_mask == 1
    color_array = np.array(color, dtype=np.uint8)
    
    # 원본 이미지 위에 빨간색 오버레이
    seg_result[mask_bool, :] = seg_result[mask_bool, :] * (1 - alpha) + color_array * alpha
    
    return seg_result

def save_detection_to_excel(detection_data, excel_path):
    """
    탐지 결과를 Excel 파일로 저장
    
    Args:
        detection_data: 탐지 데이터 리스트
        excel_path: Excel 파일 경로
    """
    if not detection_data:
        print("저장할 탐지 데이터가 없습니다.")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(detection_data, columns=['이미지 이름', '위도', '경도', '크랙 개수', '최대 크기'])
    
    # Excel 파일로 저장
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"탐지 결과가 Excel 파일에 저장되었습니다: {excel_path}")
    except Exception as e:
        print(f"Excel 파일 저장 중 오류 발생: {e}")

def extract_image_info_from_path(image_path):
    """
    이미지 경로에서 이미지 정보 추출 (예시 함수)
    실제 환경에 맞게 수정이 필요할 수 있습니다.
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        tuple: (이미지_이름, 위도, 경도)
    """
    # 기본값 설정 (실제 환경에 맞게 수정 필요)
    image_name = os.path.basename(image_path)
    
    # 예시: 파일명에서 좌표 추출하거나 메타데이터에서 읽기
    # 실제로는 이미지 EXIF 데이터나 파일명 패턴에서 추출해야 함
    default_latitude = CONFIG['DEFAULT_LATITUDE']
    default_longitude = CONFIG['DEFAULT_LONGITUDE']
    
    # 파일명 패턴으로 좌표 추출 시도 (예시)
    # 예: "IMG_20231201_37.5665_126.9780.png"
    try:
        name_parts = image_name.split('_')
        if len(name_parts) >= 4:
            lat = float(name_parts[-2])
            lon = float(name_parts[-1].split('.')[0])
            return image_name, lat, lon
    except:
        pass
    
    return image_name, default_latitude, default_longitude

def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    create_output_directories()
    
    # Excel 출력 경로 설정 (명령행 인수로 오버라이드 가능)
    excel_output_path = args.excel_output if args.excel_output else CONFIG['EXCEL_OUTPUT_PATH']
    
    # 모델 초기화
    print("크랙 탐지 모델을 초기화하는 중...")
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')
    
    # 입력 이미지 리스트 생성
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))
    print(f"처리할 이미지 개수: {len(img_list)}")
    
    # 크랙 팔레트 설정
    crack_palette = crack_model.dataset_meta['palette'][:2]
    
    if args.rgb_to_bgr:
        crack_palette = [p[::-1] for p in crack_palette]
    
    if args.overwrite_crack_palette:
        crack_palette[1] = [0, 0, 255]
    
    # 탐지 결과 저장용 리스트
    detection_results = []
    
    # 각 이미지에 대해 처리
    for idx, img_path in enumerate(track_progress(img_list, description="크랙 탐지 진행 중")):
        print(f"\n처리 중: {os.path.basename(img_path)} ({idx+1}/{len(img_list)})")
        
        try:
            # 크랙 탐지 수행
            _, crack_mask = inference_segmentor_sliding_window(
                crack_model, img_path, 
                color_mask=None, 
                score_thr=CONFIG['SCORE_THRESHOLD'], 
                window_size=CONFIG['WINDOW_SIZE'], 
                overlap_ratio=CONFIG['OVERLAP_RATIO']
            )
            
            # 원본 이미지 로드
            seg_result = mmcv.imread(img_path)
            
            # 크랙 정량화 수행 (시각화용)
            seg_result_with_measurements, crack_quantification_results = quantify_crack_width_length(
                seg_result.copy(), crack_mask, crack_palette[1]
            )
            
            # 크기 필터링 적용
            filtered_cracks = filter_crack_by_size(
                crack_quantification_results,
                min_area=CONFIG['MIN_CRACK_AREA'],
                min_width=CONFIG['MIN_CRACK_WIDTH'],
                min_length=CONFIG['MIN_CRACK_LENGTH']
            )
            
            print(f"  전체 탐지: {len(crack_quantification_results)}개, 필터링 후: {len(filtered_cracks)}개")
            
            # 필터링된 크랙이 있는 경우에만 처리
            if filtered_cracks:
                # 이미지 정보 추출
                image_name, latitude, longitude = extract_image_info_from_path(img_path)
                
                # 최대 크랙 크기 계산
                max_size = max([float(crack[1].split('x')[0]) * float(crack[1].split('x')[1]) 
                              for crack in filtered_cracks])
                
                # 탐지 결과 저장
                detection_results.append([
                    image_name,
                    latitude,
                    longitude,
                    len(filtered_cracks),
                    f"{max_size:.2f}"
                ])
                
                # 빨간색 오버레이 시각화 (정량화 텍스트 제외)
                visualized_image = visualize_crack_detection(
                    seg_result.copy(), crack_mask, 
                    color=CONFIG['CRACK_COLOR'], 
                    alpha=CONFIG['VISUALIZATION_ALPHA']
                )
                
                # 결과 이미지 저장
                rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
                mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)
                
                rst_path = os.path.join(args.rst_dir, rst_name)
                mask_path = os.path.join(args.rst_dir, mask_name)
                vis_path = os.path.join(CONFIG['IMAGE_OUTPUT_PATH'], f"visualized_{rst_name}")
                
                # 파일 저장
                mmcv.imwrite(visualized_image, rst_path)
                mmcv.imwrite(crack_mask.astype(np.uint8), mask_path)
                mmcv.imwrite(visualized_image, vis_path)
                
                print(f"  결과 저장: {rst_path}")
                print(f"  시각화 저장: {vis_path}")
            
            # GPU 메모리 정리
            empty_cache()
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue
    
    # Excel 파일로 탐지 결과 저장
    if detection_results:
        save_detection_to_excel(detection_results, excel_output_path)
        print(f"\n총 {len(detection_results)}개의 이미지에서 크랙이 탐지되어 Excel에 저장되었습니다.")
    else:
        print("\n크기가 충분한 크랙이 탐지되지 않았습니다.")
    
    print("크랙 탐지 및 분석이 완료되었습니다.")

if __name__ == '__main__':
    main()
