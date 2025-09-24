# Crack Detection Inference

이 프로젝트는 고해상도 이미지에서 크랙을 탐지하고 정량화하는 Python 스크립트입니다.

## 주요 기능

- 슬라이딩 윈도우를 사용한 고해상도 이미지 크랙 탐지
- 크랙의 폭과 길이 정량화
- 시각화된 결과 이미지 생성
- CSV 형태의 정량화 결과 저장

## 파일 구조

```
prototyping/
├── inference.py              # 메인 추론 스크립트
├── utils.py                  # 슬라이딩 윈도우 유틸리티
├── quantify_seg_results.py   # 크랙 정량화 함수
├── requirements.txt          # Python 패키지 의존성
├── environment.yml           # Conda 환경 설정
├── setup.sh                  # 환경 설정 스크립트
├── run_inference.sh          # 추론 실행 스크립트
└── test_data/                # 테스트 데이터
    ├── leftImg8bit/test/     # 입력 이미지
    └── inference_results/    # 결과 이미지
```

## 설치 방법

### 1. Conda 환경 생성 및 설정

```bash
# 환경 설정 스크립트 실행
./setup.sh
```

또는 수동으로:

```bash
# Conda 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate hard
```

### 2. 의존성 확인

```bash
# 환경 활성화 후 의존성 확인
conda activate hard
python -c "import mmcv, mmseg, mmengine, cv2, numpy, pandas, skimage, slidingwindow; print('All dependencies installed successfully!')"
```

## 사용 방법

### 1. 스크립트를 사용한 실행

```bash
# 환경 활성화
conda activate hard

# 추론 실행
./run_inference.sh <config_path> <checkpoint_path> <input_dir> <output_dir>
```

예시:
```bash
./run_inference.sh /path/to/config.py /path/to/checkpoint.pth ./test_data/leftImg8bit/test ./test_data/inference_results
```

### 2. 직접 Python 스크립트 실행

```bash
# 환경 활성화
conda activate hard

# 추론 실행
python inference.py \
    --crack_config /path/to/config.py \
    --crack_checkpoint /path/to/checkpoint.pth \
    --srx_dir /path/to/input/images \
    --rst_dir /path/to/output/results
```

## 매개변수 설명

- `--crack_config`: 크랙 탐지 모델의 설정 파일 경로
- `--crack_checkpoint`: 훈련된 모델의 체크포인트 파일 경로
- `--srx_dir`: 입력 이미지가 있는 디렉토리
- `--rst_dir`: 결과를 저장할 디렉토리
- `--srx_suffix`: 입력 이미지 확장자 (기본값: .png)
- `--rst_suffix`: 결과 이미지 확장자 (기본값: .png)
- `--mask_suffix`: 마스크 이미지 확장자 (기본값: .png)
- `--alpha`: 블렌딩 투명도 (기본값: 0.8)
- `--rgb_to_bgr`: RGB를 BGR로 변환 (기본값: False)
- `--overwrite_crack_palette`: 크랙 팔레트를 빨간색으로 덮어쓰기 (기본값: False)
- `--scaling_factor`: 이미지 스케일링 팩터 (기본값: 1)

## 출력 결과

1. **시각화된 이미지**: 원본 이미지에 크랙 탐지 결과가 오버레이된 이미지
2. **마스크 이미지**: 크랙 영역만 표시된 이진 마스크
3. **CSV 파일**: 각 크랙의 정량화 결과 (`all_crack_quantification_results.csv`)

## 시스템 요구사항

- Python 3.9
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 충분한 디스크 공간 (입력 이미지 크기에 따라)

## 문제 해결

### 1. CUDA 관련 오류
- GPU가 없는 경우 `device='cuda:0'`을 `device='cpu'`로 변경
- CUDA 버전과 PyTorch 버전 호환성 확인

### 2. 메모리 부족 오류
- `window_size` 매개변수를 줄여서 슬라이딩 윈도우 크기 감소
- 이미지 크기를 줄여서 처리

### 3. 의존성 설치 오류
- Conda 환경을 새로 생성: `conda env remove -n hard && conda env create -f environment.yml`
- pip로 개별 설치: `pip install -r requirements.txt`

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

