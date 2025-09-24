# Enhanced Crack Detection System
## 개선된 크랙 탐지 시스템

기존 크랙 탐지 시스템에 크기 필터링, Excel 내보내기, 시각화 기능을 추가한 개선된 시스템입니다.

## 🚀 주요 기능

### 1. 크랙 탐지
- MMSegmentation 기반 딥러닝 모델을 사용한 정확한 크랙 탐지
- 슬라이딩 윈도우 방식으로 대용량 이미지 처리

### 2. 크기 필터링
- 특정 크기 이하의 탐지 결과 자동 무시
- 최소 면적, 폭, 길이 기준으로 필터링
- 설정 파일에서 임계값 조정 가능

### 3. Excel 내보내기
- 탐지된 크랙이 있는 이미지 정보를 Excel 파일로 자동 저장
- 이미지 이름, 위도, 경도, 크랙 개수, 최대 크기 정보 포함

### 4. 시각화
- 빨간색 오버레이로 크랙 위치 시각화
- 정량화 텍스트 제거하여 깔끔한 시각화
- 투명도 조정 가능

## 📁 파일 구조

```
inferences/
├── enhanced_crack_inference.py    # 메인 추론 스크립트
├── config.py                      # 설정 파일
├── run_enhanced_crack_detection.sh # 실행 스크립트
├── quantify_seg_results.py        # 크랙 정량화 모듈
├── utils.py                       # 슬라이딩 윈도우 유틸리티
└── README_Enhanced.md             # 이 파일
```

## ⚙️ 설정

### config.py에서 수정 가능한 설정들

```python
# 경로 설정
EXCEL_OUTPUT_PATH = "/path/to/crack_results.xlsx"
IMAGE_OUTPUT_PATH = "/path/to/output/images"

# 크기 필터링 임계값
MIN_CRACK_AREA = 1000      # 최소 크랙 면적 (픽셀)
MIN_CRACK_WIDTH = 3.0      # 최소 크랙 폭 (픽셀)
MIN_CRACK_LENGTH = 50.0    # 최소 크랙 길이 (픽셀)

# 시각화 설정
VISUALIZATION_ALPHA = 0.8  # 투명도 (0.0-1.0)
CRACK_COLOR = [0, 0, 255]  # 빨간색 (BGR 형식)

# 모델 설정
WINDOW_SIZE = 1024         # 슬라이딩 윈도우 크기
OVERLAP_RATIO = 0.1        # 윈도우 겹침 비율
SCORE_THRESHOLD = 0.1      # 신뢰도 임계값
```

## 🎯 사용법

### 1. 기본 실행

```bash
python enhanced_crack_inference.py \
    --crack_config "path/to/config.py" \
    --crack_checkpoint "path/to/checkpoint.pth" \
    --srx_dir "path/to/input/images" \
    --rst_dir "path/to/output"
```

### 2. 스크립트 실행

```bash
./run_enhanced_crack_detection.sh
```

### 3. 추가 옵션

```bash
python enhanced_crack_inference.py \
    --crack_config "config.py" \
    --crack_checkpoint "checkpoint.pth" \
    --srx_dir "input_dir" \
    --rst_dir "output_dir" \
    --excel_output "custom_results.xlsx" \
    --srx_suffix ".jpg" \
    --rst_suffix ".png" \
    --mask_suffix ".png" \
    --rgb_to_bgr
```

## 📊 출력 결과

### 1. Excel 파일
탐지된 크랙이 있는 이미지들의 정보가 저장됩니다:

| 이미지 이름 | 위도 | 경도 | 크랙 개수 | 최대 크기 |
|------------|------|------|-----------|-----------|
| image1.png | 37.5665 | 126.9780 | 3 | 150.25 |
| image2.png | 37.5666 | 126.9781 | 1 | 75.50 |

### 2. 이미지 파일
- **원본 + 크랙 표시**: 빨간색 오버레이로 크랙 위치 표시
- **마스크**: 크랙 영역만 표시된 이진 마스크
- **시각화**: 정량화 텍스트가 제거된 깔끔한 시각화

## 🔧 커스터마이징

### 좌표 추출 함수 수정
`extract_image_info_from_path()` 함수를 수정하여 실제 환경에 맞는 좌표 추출 로직을 구현할 수 있습니다.

### 크기 필터링 기준 조정
`config.py`에서 `MIN_CRACK_AREA`, `MIN_CRACK_WIDTH`, `MIN_CRACK_LENGTH` 값을 조정하여 탐지 기준을 변경할 수 있습니다.

### 시각화 스타일 변경
`config.py`에서 `CRACK_COLOR`, `VISUALIZATION_ALPHA` 값을 조정하여 시각화 스타일을 변경할 수 있습니다.

## 📝 주의사항

1. **좌표 정보**: 현재는 기본 좌표(서울시청)를 사용합니다. 실제 환경에서는 이미지 EXIF 데이터나 파일명 패턴에서 좌표를 추출하도록 `extract_image_info_from_path()` 함수를 수정해야 합니다.

2. **메모리 관리**: 대용량 이미지 처리 시 GPU 메모리가 부족할 수 있습니다. `WINDOW_SIZE`를 줄이거나 배치 크기를 조정하세요.

3. **파일 형식**: 입력 이미지는 PNG, JPG 등 일반적인 형식을 지원합니다. `srx_suffix`로 확장자를 지정하세요.

## 🐛 문제 해결

### GPU 메모리 부족
```python
# config.py에서 윈도우 크기 줄이기
WINDOW_SIZE = 512  # 1024에서 512로 감소
```

### 탐지 결과가 너무 많음
```python
# config.py에서 필터링 기준 강화
MIN_CRACK_AREA = 2000    # 면적 기준 강화
MIN_CRACK_WIDTH = 5.0    # 폭 기준 강화
MIN_CRACK_LENGTH = 100.0 # 길이 기준 강화
```

### 탐지 결과가 너무 적음
```python
# config.py에서 필터링 기준 완화
MIN_CRACK_AREA = 500     # 면적 기준 완화
MIN_CRACK_WIDTH = 2.0    # 폭 기준 완화
MIN_CRACK_LENGTH = 25.0  # 길이 기준 완화
```
