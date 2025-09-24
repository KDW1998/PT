#!/bin/bash

# Enhanced Crack Detection 실행 스크립트
# 개선된 크랙 탐지 시스템 실행 예제

echo "=== Enhanced Crack Detection System ==="
echo "개선된 크랙 탐지 시스템을 시작합니다..."

# 설정값 (필요에 따라 수정)
CRACK_CONFIG="/home/deogwonkang/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/24.09.12_only_crack/convnext_tiny_fpn_crack_hardnegative_100units.py"
CRACK_CHECKPOINT="/home/deogwonkang/WindowsShare/05. Data/03. Checkpoints/hardnegative/학습데이터_방법_및_데이터개수로_분리/24.09.12_only_crack/iter_best.pth"
INPUT_DIR="/home/deogwonkang/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/테스트데이터/br테스트데이터/대표테스트데이터/leftImg8bit/test"
OUTPUT_DIR="/home/deogwonkang/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/탐지결과_비교/Enhanced/Joint"

# Excel 출력 경로 (선택사항)
EXCEL_OUTPUT="/home/deogwonkang/Eunji_MapDemo/PyDracula/init/data/enhanced_crack_results.xlsx"

# 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "입력 디렉토리: $INPUT_DIR"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "Excel 출력: $EXCEL_OUTPUT"
echo ""

# Enhanced Crack Detection 실행
python enhanced_crack_inference.py \
    --crack_config "$CRACK_CONFIG" \
    --crack_checkpoint "$CRACK_CHECKPOINT" \
    --srx_dir "$INPUT_DIR" \
    --rst_dir "$OUTPUT_DIR" \
    --excel_output "$EXCEL_OUTPUT" \
    --srx_suffix ".png" \
    --rst_suffix ".png" \
    --mask_suffix ".png" \
    --rgb_to_bgr

echo ""
echo "=== 크랙 탐지 완료 ==="
echo "결과 파일들을 확인하세요:"
echo "- 탐지 결과 이미지: $OUTPUT_DIR"
echo "- Excel 결과 파일: $EXCEL_OUTPUT"
echo "- 시각화 이미지: $(python -c "from config import CONFIG; print(CONFIG['IMAGE_OUTPUT_PATH'])")"
