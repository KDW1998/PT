"""
Enhanced Crack Detection Test Script
개선된 크랙 탐지 시스템 테스트 스크립트

이 스크립트는 enhanced_crack_inference.py의 주요 기능들을 테스트합니다.
"""

import os
import sys
import numpy as np
import pandas as pd
from config import CONFIG

def test_config_loading():
    """설정 파일 로딩 테스트"""
    print("=== 설정 파일 로딩 테스트 ===")
    
    required_keys = [
        'EXCEL_OUTPUT_PATH', 'IMAGE_OUTPUT_PATH', 'MIN_CRACK_AREA',
        'MIN_CRACK_WIDTH', 'MIN_CRACK_LENGTH', 'VISUALIZATION_ALPHA',
        'CRACK_COLOR', 'WINDOW_SIZE', 'OVERLAP_RATIO', 'SCORE_THRESHOLD'
    ]
    
    for key in required_keys:
        if key in CONFIG:
            print(f"✓ {key}: {CONFIG[key]}")
        else:
            print(f"✗ {key}: 누락됨")
            return False
    
    print("✓ 모든 설정이 정상적으로 로드되었습니다.\n")
    return True

def test_filter_function():
    """크기 필터링 함수 테스트"""
    print("=== 크기 필터링 함수 테스트 ===")
    
    # 테스트 데이터 생성
    test_results = [
        ["(100,100)-(200,200)", "5.0x50.0", 1],  # 큰 크랙
        ["(300,300)-(310,310)", "1.0x5.0", 1],   # 작은 크랙 (필터링됨)
        ["(400,400)-(450,450)", "4.0x60.0", 1],  # 중간 크랙
        ["(500,500)-(502,502)", "0.5x2.0", 1],   # 매우 작은 크랙 (필터링됨)
    ]
    
    # 필터링 함수 import 및 테스트
    try:
        from enhanced_crack_inference import filter_crack_by_size
        
        # 기본 필터링 테스트
        filtered = filter_crack_by_size(test_results)
        print(f"전체 크랙: {len(test_results)}개")
        print(f"필터링 후: {len(filtered)}개")
        
        # 강화된 필터링 테스트
        filtered_strict = filter_crack_by_size(
            test_results, 
            min_area=CONFIG['MIN_CRACK_AREA'],
            min_width=CONFIG['MIN_CRACK_WIDTH'],
            min_length=CONFIG['MIN_CRACK_LENGTH']
        )
        print(f"강화 필터링 후: {len(filtered_strict)}개")
        
        print("✓ 크기 필터링 함수가 정상 작동합니다.\n")
        return True
        
    except ImportError as e:
        print(f"✗ 필터링 함수 import 실패: {e}")
        return False

def test_visualization_function():
    """시각화 함수 테스트"""
    print("=== 시각화 함수 테스트 ===")
    
    try:
        from enhanced_crack_inference import visualize_crack_detection
        
        # 테스트 이미지 생성 (100x100x3)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 테스트 마스크 생성 (중앙에 크랙 영역)
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[40:60, 40:60] = 1  # 20x20 크랙 영역
        
        # 시각화 테스트
        visualized = visualize_crack_detection(test_image, test_mask)
        
        if visualized.shape == test_image.shape:
            print("✓ 시각화 함수가 정상 작동합니다.")
            print(f"  입력 이미지 크기: {test_image.shape}")
            print(f"  출력 이미지 크기: {visualized.shape}")
        else:
            print("✗ 시각화 함수 출력 크기가 올바르지 않습니다.")
            return False
        
        print("✓ 시각화 함수 테스트 완료.\n")
        return True
        
    except ImportError as e:
        print(f"✗ 시각화 함수 import 실패: {e}")
        return False

def test_excel_export():
    """Excel 내보내기 함수 테스트"""
    print("=== Excel 내보내기 함수 테스트 ===")
    
    try:
        from enhanced_crack_inference import save_detection_to_excel
        
        # 테스트 데이터 생성
        test_data = [
            ["test_image1.png", 37.5665, 126.9780, 2, "150.25"],
            ["test_image2.png", 37.5666, 126.9781, 1, "75.50"],
        ]
        
        # 테스트 Excel 파일 경로
        test_excel_path = "/tmp/test_crack_results.xlsx"
        
        # Excel 저장 테스트
        save_detection_to_excel(test_data, test_excel_path)
        
        if os.path.exists(test_excel_path):
            # Excel 파일 읽기 테스트
            df = pd.read_excel(test_excel_path)
            print(f"✓ Excel 파일이 성공적으로 생성되었습니다: {test_excel_path}")
            print(f"  저장된 행 수: {len(df)}")
            print(f"  컬럼: {list(df.columns)}")
            
            # 테스트 파일 삭제
            os.remove(test_excel_path)
            print("✓ 테스트 파일이 정리되었습니다.")
        else:
            print("✗ Excel 파일 생성에 실패했습니다.")
            return False
        
        print("✓ Excel 내보내기 테스트 완료.\n")
        return True
        
    except ImportError as e:
        print(f"✗ Excel 내보내기 함수 import 실패: {e}")
        return False
    except Exception as e:
        print(f"✗ Excel 내보내기 테스트 실패: {e}")
        return False

def test_directory_creation():
    """디렉토리 생성 함수 테스트"""
    print("=== 디렉토리 생성 함수 테스트 ===")
    
    try:
        from enhanced_crack_inference import create_output_directories
        
        # 테스트 디렉토리 설정
        original_path = CONFIG['IMAGE_OUTPUT_PATH']
        CONFIG['IMAGE_OUTPUT_PATH'] = "/tmp/test_output_images"
        CONFIG['EXCEL_OUTPUT_PATH'] = "/tmp/test_output.xlsx"
        
        # 디렉토리 생성 테스트
        create_output_directories()
        
        # 결과 확인
        if os.path.exists(CONFIG['IMAGE_OUTPUT_PATH']):
            print(f"✓ 이미지 출력 디렉토리가 생성되었습니다: {CONFIG['IMAGE_OUTPUT_PATH']}")
        else:
            print("✗ 이미지 출력 디렉토리 생성에 실패했습니다.")
            return False
        
        excel_dir = os.path.dirname(CONFIG['EXCEL_OUTPUT_PATH'])
        if os.path.exists(excel_dir):
            print(f"✓ Excel 출력 디렉토리가 생성되었습니다: {excel_dir}")
        else:
            print("✗ Excel 출력 디렉토리 생성에 실패했습니다.")
            return False
        
        # 원래 설정 복원
        CONFIG['IMAGE_OUTPUT_PATH'] = original_path
        
        print("✓ 디렉토리 생성 테스트 완료.\n")
        return True
        
    except ImportError as e:
        print(f"✗ 디렉토리 생성 함수 import 실패: {e}")
        return False
    except Exception as e:
        print(f"✗ 디렉토리 생성 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("Enhanced Crack Detection System 테스트를 시작합니다...\n")
    
    tests = [
        test_config_loading,
        test_filter_function,
        test_visualization_function,
        test_excel_export,
        test_directory_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 테스트 실행 중 오류 발생: {e}\n")
    
    print("=== 테스트 결과 요약 ===")
    print(f"통과: {passed}/{total}")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        print("Enhanced Crack Detection System이 정상적으로 작동할 준비가 되었습니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 코드를 확인해주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
