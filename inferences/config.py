"""
Enhanced Crack Detection Configuration
개선된 크랙 탐지 시스템 설정 파일

이 파일에서 모든 설정값을 수정할 수 있습니다.
"""

import os

# =============================================================================
# 기본 경로 설정
# =============================================================================
BASE_DIR = "/home/user/PT"

# Excel 파일 저장 경로 (수정 가능)
EXCEL_OUTPUT_PATH = os.path.join(BASE_DIR, "PyDracula/init/data/crack_detection_results.xlsx")

# 탐지 결과 이미지 저장 경로 (수정 가능)
IMAGE_OUTPUT_PATH = os.path.join(BASE_DIR, "crack_detection_images")

# =============================================================================
# 크기 필터링 설정 (픽셀 단위)
# =============================================================================
# 최소 크랙 면적 (픽셀 단위)
MIN_CRACK_AREA = 0

# 최소 크랙 폭 (픽셀 단위)
MIN_CRACK_WIDTH = 0

# 최소 크랙 길이 (픽셀 단위)
MIN_CRACK_LENGTH = 0

# =============================================================================
# 시각화 설정
# =============================================================================
# 빨간색 오버레이 투명도 (0.0-1.0)
VISUALIZATION_ALPHA = 0.8

# 크랙 표시 색상 (BGR 형식)
CRACK_COLOR = [0, 0, 255]  # 빨간색

# =============================================================================
# 모델 추론 설정
# =============================================================================
# 슬라이딩 윈도우 크기
WINDOW_SIZE = 1024

# 윈도우 겹침 비율
OVERLAP_RATIO = 0.5

# 신뢰도 임계값
SCORE_THRESHOLD = 0.5

# =============================================================================
# 기본 좌표 설정 (이미지에서 좌표를 추출할 수 없는 경우 사용)
# =============================================================================
DEFAULT_LATITUDE = 37.5665   # 서울시청 위도
DEFAULT_LONGITUDE = 126.9780  # 서울시청 경도

# =============================================================================
# 파일 확장자 설정
# =============================================================================
DEFAULT_INPUT_SUFFIX = '.png'
DEFAULT_OUTPUT_SUFFIX = '.png'
DEFAULT_MASK_SUFFIX = '.png'

# =============================================================================
# 설정 사전 정의 (코드에서 사용)
# =============================================================================
CONFIG = {
    'EXCEL_OUTPUT_PATH': EXCEL_OUTPUT_PATH,
    'IMAGE_OUTPUT_PATH': IMAGE_OUTPUT_PATH,
    'MIN_CRACK_AREA': MIN_CRACK_AREA,
    'MIN_CRACK_WIDTH': MIN_CRACK_WIDTH,
    'MIN_CRACK_LENGTH': MIN_CRACK_LENGTH,
    'VISUALIZATION_ALPHA': VISUALIZATION_ALPHA,
    'CRACK_COLOR': CRACK_COLOR,
    'WINDOW_SIZE': WINDOW_SIZE,
    'OVERLAP_RATIO': OVERLAP_RATIO,
    'SCORE_THRESHOLD': SCORE_THRESHOLD,
    'DEFAULT_LATITUDE': DEFAULT_LATITUDE,
    'DEFAULT_LONGITUDE': DEFAULT_LONGITUDE,
    'DEFAULT_INPUT_SUFFIX': DEFAULT_INPUT_SUFFIX,
    'DEFAULT_OUTPUT_SUFFIX': DEFAULT_OUTPUT_SUFFIX,
    'DEFAULT_MASK_SUFFIX': DEFAULT_MASK_SUFFIX,
}
