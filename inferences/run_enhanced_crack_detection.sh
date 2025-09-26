#!/bin/bash

# Enhanced Crack Detection 실행 스크립트
# 개선된 크랙 탐지 시스템 실행 예제

# # chmod +x inferences/run_enhanced_crack_detection.sh
# # bash inferences/run_enhanced_crack_detection.sh

echo "=== Enhanced Crack Detection System ==="
echo "개선된 크랙 탐지 시스템을 시작합니다..."

# 설정값 (필요에 따라 수정)
CRACK_CONFIG="/home/user/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/Only_Positive/seg_PSOnly_Positive800_noOversampling/convnext_tiny_fpn_crack_hardnegative_100units.py"
CRACK_CHECKPOINT="/home/user/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/Only_Positive/seg_PSOnly_Positive800_noOversampling/iter_best.pth"
INPUT_DIR="/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/genenral_crack_v0.1.2/leftImg8bit/train"
OUTPUT_DIR="/home/user/PT/Examples"

# Excel 출력 경로 (선택사항)
EXCEL_OUTPUT="/home/user/PT/Examples/enhanced_crack_results.xlsx"

# 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "입력 디렉토리: $INPUT_DIR"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "Excel 출력: $EXCEL_OUTPUT"
echo ""

# Enhanced Crack Detection 실행
python inferences/enhanced_crack_inference.py \
    --crack_config "$CRACK_CONFIG" \
    --crack_checkpoint "$CRACK_CHECKPOINT" \
    --srx_dir "$INPUT_DIR" \
    --rst_dir "$OUTPUT_DIR" \
    --excel_output "$EXCEL_OUTPUT" \
    --srx_suffix ".png" \
    --rst_suffix ".jpg" \
    --mask_suffix ".jpg" \
    --rgb_to_bgr

echo ""
echo "=== 크랙 탐지 완료 ==="
echo "결과 파일들을 확인하세요:"
echo "- 탐지 결과 이미지: $OUTPUT_DIR"
echo "- Excel 결과 파일: $EXCEL_OUTPUT"
echo "- 시각화 이미지: $(python -c "import sys; sys.path.append('inferences'); from config import CONFIG; print(CONFIG['IMAGE_OUTPUT_PATH'])")"

echo ""
echo "=== 마스크 시각화 생성 중 ==="
echo "원본 이미지에 반투명 빨간색 오버레이를 생성합니다..."

# 시각화 출력 디렉토리 설정
VISUALIZATION_DIR="$OUTPUT_DIR/visualization"

# 마스크 시각화 실행
python inferences/overlay_and_save_gt.py \
    --img_dir "$INPUT_DIR" \
    --gt_dir "$OUTPUT_DIR" \
    --output_dir "$VISUALIZATION_DIR" \
    --target_label 1 \
    --alpha 0.6 \
    --img_suffix ".JPG" \
    --gt_suffix ".jpg"

echo ""
echo "=== 마스크 시각화 완료 ==="
echo "시각화 결과를 확인하세요:"
echo "- 시각화 이미지: $VISUALIZATION_DIR"
