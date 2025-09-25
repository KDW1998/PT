#!/usr/bin/env python3
"""
Test script for new Excel output format
새로운 Excel 출력 형식 테스트 스크립트

This script demonstrates the new Excel format with latitude, longitude, and image path.
"""

import pandas as pd
import os

def test_excel_format():
    """Test the new Excel output format"""
    
    print("=== Excel Output Format Test ===")
    
    # Sample data in the new format
    sample_data = [
        [35.0339588, 126.5028282, "/path/to/images/202508290304104386.jpg"],
        [35.0323817, 126.4987719, "/path/to/images/202508290304051120.jpg"],
        [35.0320539, 126.4976715, "/path/to/images/202508290304003370.jpg"],
    ]
    
    # Create DataFrame with new column names
    df = pd.DataFrame(sample_data, columns=['위도', '경도', '이미지 경로'])
    
    print("새로운 Excel 형식:")
    print(df.to_string(index=False))
    print()
    
    # Test saving to Excel
    test_excel_path = "/home/user/PT/test_output.xlsx"
    try:
        df.to_excel(test_excel_path, index=False, engine='openpyxl')
        print(f"테스트 Excel 파일이 저장되었습니다: {test_excel_path}")
        
        # Verify the saved file
        if os.path.exists(test_excel_path):
            print("✓ Excel 파일이 성공적으로 생성되었습니다.")
            # Clean up test file
            os.remove(test_excel_path)
            print("✓ 테스트 파일이 정리되었습니다.")
        else:
            print("✗ Excel 파일 생성에 실패했습니다.")
            
    except Exception as e:
        print(f"Excel 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    test_excel_format()
