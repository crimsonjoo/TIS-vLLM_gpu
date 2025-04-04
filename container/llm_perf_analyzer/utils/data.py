#!/usr/bin/env python3
"""
GenAI 성능 테스트 결과 데이터 처리 모듈

이 모듈은 GenAI-Perf 성능 테스트 결과 데이터를 처리하는 함수들을 제공합니다.
CSV 파일 파싱, 데이터 전처리, 결과 파일 관리 등의 기능을 담당합니다.

주요 기능:
- CSV 파일 파싱 및 데이터프레임 변환
- 결과 파일 검색 및 관리
- 데이터 전처리 및 정제
"""

import os
import glob
import pandas as pd
import numpy as np
import json

def parse_csv_file(csv_file):
    """
    CSV 파일을 파싱하여 데이터프레임으로 변환합니다.
    
    Args:
        csv_file (str): CSV 파일 경로
        
    Returns:
        DataFrame: 파싱된 데이터프레임
    """
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"CSV 파일 파싱 오류: {e}")
        return None

def find_result_files(result_dir):
    """
    결과 디렉토리에서 CSV 파일들을 찾습니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        dict: 파일 유형별 경로 딕셔너리
    """
    result_files = {}
    
    # 지표 결과 파일
    metrics_file = os.path.join(result_dir, "metrics.csv")
    if os.path.exists(metrics_file):
        result_files["metrics"] = metrics_file
    
    # 지연 시간 결과 파일
    latency_file = os.path.join(result_dir, "latency.csv")
    if os.path.exists(latency_file):
        result_files["latency"] = latency_file
    
    # GPU 메트릭 결과 파일
    gpu_metrics_file = os.path.join(result_dir, "gpu_metrics.csv")
    if os.path.exists(gpu_metrics_file):
        result_files["gpu_metrics"] = gpu_metrics_file
    
    # 요약 결과 파일
    summary_file = os.path.join(result_dir, "summary.csv")
    if os.path.exists(summary_file):
        result_files["summary"] = summary_file
    
    return result_files

def load_result_data(result_dir):
    """
    결과 디렉토리에서 모든 데이터를 로드합니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        dict: 데이터 유형별 데이터프레임 딕셔너리
    """
    result_files = find_result_files(result_dir)
    result_data = {}
    
    for data_type, file_path in result_files.items():
        df = parse_csv_file(file_path)
        if df is not None:
            result_data[data_type] = df
    
    return result_data

def get_test_info(result_dir):
    """
    테스트 정보를 가져옵니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        dict: 테스트 정보 딕셔너리
    """
    info_file = os.path.join(result_dir, "info.json")
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"테스트 정보 파일 파싱 오류: {e}")
    
    # info.json이 없는 경우 디렉토리 이름에서 정보 추출
    dir_name = os.path.basename(result_dir)
    parts = dir_name.split('_')
    
    info = {
        "model": parts[1] if len(parts) > 1 else "unknown",
        "timestamp": parts[2] if len(parts) > 2 else "unknown",
    }
    
    return info

def find_result_dirs(artifacts_dir):
    """
    아티팩트 디렉토리에서 모든 결과 디렉토리를 찾습니다.
    
    Args:
        artifacts_dir (str): 아티팩트 디렉토리 경로
        
    Returns:
        list: 결과 디렉토리 경로 리스트
    """
    result_dirs = []
    
    # perf_로 시작하는 모든 디렉토리 찾기
    pattern = os.path.join(artifacts_dir, "perf_*")
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            result_dirs.append(dir_path)
    
    # 타임스탬프 기준으로 정렬 (최신 순)
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return result_dirs

def preprocess_metrics_data(df):
    """
    지표 데이터를 전처리합니다.
    
    Args:
        df (DataFrame): 원본 데이터프레임
        
    Returns:
        DataFrame: 전처리된 데이터프레임
    """
    if df is None or df.empty:
        return df
    
    # 숫자 컬럼 변환
    numeric_cols = ['avg', 'min', 'max', 'p25', 'p50', 'p75', 'p90', 'p99']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
    
    return df

def get_model_name(result_dir):
    """
    결과 디렉토리에서 모델 이름을 추출합니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        str: 모델 이름
    """
    dir_name = os.path.basename(result_dir)
    parts = dir_name.split('_')
    
    if len(parts) > 1:
        return parts[1]
    
    return "unknown"

def get_timestamp(result_dir):
    """
    결과 디렉토리에서 타임스탬프를 추출합니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        str: 타임스탬프
    """
    dir_name = os.path.basename(result_dir)
    parts = dir_name.split('_')
    
    if len(parts) > 2:
        return parts[2]
    
    return "unknown"

def filter_results_by_model(result_dirs, model_name):
    """
    모델 이름으로 결과 디렉토리를 필터링합니다.
    
    Args:
        result_dirs (list): 결과 디렉토리 경로 리스트
        model_name (str): 모델 이름
        
    Returns:
        list: 필터링된 결과 디렉토리 경로 리스트
    """
    filtered_dirs = []
    
    for dir_path in result_dirs:
        dir_model = get_model_name(dir_path)
        if dir_model.lower() == model_name.lower():
            filtered_dirs.append(dir_path)
    
    return filtered_dirs
