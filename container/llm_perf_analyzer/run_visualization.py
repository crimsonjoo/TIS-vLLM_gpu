#!/usr/bin/env python3
"""
GenAI 성능 테스트 결과 시각화 실행 스크립트

이 스크립트는 GenAI-Perf 성능 테스트 결과를 시각화하는 유틸리티를 실행합니다.
명령줄 인수를 통해 다양한 시각화 옵션을 제공합니다.

사용법:
    python run_visualization.py --result_dir <결과_디렉토리> [옵션]
    python run_visualization.py --compare <결과_디렉토리1> <결과_디렉토리2> [옵션]

주요 기능:
- 단일 테스트 결과 시각화
- 여러 테스트 결과 비교 시각화
- 다양한 차트 유형 생성
- 결과 저장 및 출력
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 유틸리티 모듈 가져오기
from utils.data import (
    parse_csv_file, 
    find_result_files, 
    load_result_data, 
    get_test_info,
    find_result_dirs,
    preprocess_metrics_data,
    get_model_name,
    get_timestamp
)
from utils.charts import (
    create_latency_insights_chart,
    create_throughput_insights_chart,
    create_bar_chart,
    create_percentile_chart,
    create_gpu_metrics_chart
)
from utils.compare import (
    compare_results,
    compare_gpu_metrics
)

def visualize_single_result(result_dir, output_dir=None, chart_types=None):
    """
    단일 테스트 결과를 시각화합니다.
    
    Args:
        result_dir (str): 결과 디렉토리 경로
        output_dir (str, optional): 출력 디렉토리 경로. 기본값은 None.
        chart_types (list, optional): 생성할 차트 유형 리스트. 기본값은 None.
        
    Returns:
        bool: 성공 여부
    """
    # 결과 데이터 로드
    result_data = load_result_data(result_dir)
    if not result_data:
        print(f"오류: {result_dir}에서 결과 데이터를 찾을 수 없습니다.")
        return False
    
    # 테스트 정보 가져오기
    test_info = get_test_info(result_dir)
    model_name = test_info.get("model", "unknown")
    timestamp = test_info.get("timestamp", "unknown")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(result_dir, "charts")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본 차트 유형 설정
    if chart_types is None:
        chart_types = ["latency", "throughput", "percentile", "gpu"]
    
    success = False
    
    # 요약 데이터 전처리
    summary_df = None
    if "summary" in result_data:
        summary_df = preprocess_metrics_data(result_data["summary"])
    
    # 지연 시간 인사이트 차트
    if "latency" in chart_types and summary_df is not None:
        output_file = os.path.join(output_dir, f"{model_name}_latency_insights.png")
        title = f"{model_name} 지연 시간 인사이트"
        if create_latency_insights_chart(summary_df, title, output_file):
            print(f"지연 시간 인사이트 차트가 생성되었습니다: {output_file}")
            success = True
    
    # 처리량 인사이트 차트
    if "throughput" in chart_types and summary_df is not None:
        output_file = os.path.join(output_dir, f"{model_name}_throughput_insights.png")
        title = f"{model_name} 처리량 인사이트"
        if create_throughput_insights_chart(summary_df, title, output_file):
            print(f"처리량 인사이트 차트가 생성되었습니다: {output_file}")
            success = True
    
    # 백분위수 차트
    if "percentile" in chart_types and summary_df is not None:
        output_file = os.path.join(output_dir, f"{model_name}_percentile.png")
        title = f"{model_name} 지연 시간 백분위수"
        if create_percentile_chart(summary_df, title, output_file):
            print(f"백분위수 차트가 생성되었습니다: {output_file}")
            success = True
    
    # GPU 메트릭 차트
    if "gpu" in chart_types and "gpu_metrics" in result_data:
        gpu_df = preprocess_metrics_data(result_data["gpu_metrics"])
        output_file = os.path.join(output_dir, f"{model_name}_gpu_metrics.png")
        title = f"{model_name} GPU 메트릭"
        if create_gpu_metrics_chart(gpu_df, title, output_file):
            print(f"GPU 메트릭 차트가 생성되었습니다: {output_file}")
            success = True
    
    return success

def compare_multiple_results(result_dirs, output_dir=None, chart_types=None, names=None):
    """
    여러 테스트 결과를 비교 시각화합니다.
    
    Args:
        result_dirs (list): 결과 디렉토리 경로 리스트
        output_dir (str, optional): 출력 디렉토리 경로. 기본값은 None.
        chart_types (list, optional): 생성할 차트 유형 리스트. 기본값은 None.
        names (list, optional): 각 결과의 이름 리스트. 기본값은 None.
        
    Returns:
        bool: 성공 여부
    """
    if len(result_dirs) < 2:
        print("오류: 비교를 위해서는 최소 2개의 결과 디렉토리가 필요합니다.")
        return False
    
    # 결과 데이터 로드
    result_dfs = []
    model_names = []
    
    for result_dir in result_dirs:
        result_data = load_result_data(result_dir)
        if not result_data or "summary" not in result_data:
            print(f"오류: {result_dir}에서 요약 데이터를 찾을 수 없습니다.")
            continue
        
        summary_df = preprocess_metrics_data(result_data["summary"])
        result_dfs.append(summary_df)
        
        model_name = get_model_name(result_dir)
        model_names.append(model_name)
    
    if len(result_dfs) < 2:
        print("오류: 비교를 위한 충분한 데이터를 찾을 수 없습니다.")
        return False
    
    # 이름 설정
    if names is None:
        names = model_names
    
    # 출력 디렉토리 설정
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(result_dirs[0]), f"compare_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본 차트 유형 설정
    if chart_types is None:
        chart_types = ["performance", "gpu"]
    
    success = False
    
    # 성능 비교 차트
    if "performance" in chart_types:
        output_file = os.path.join(output_dir, "performance_comparison.png")
        title = "성능 비교"
        if compare_results(result_dfs, names, title, output_file):
            print(f"성능 비교 차트가 생성되었습니다: {output_file}")
            success = True
    
    # GPU 메트릭 비교 차트
    if "gpu" in chart_types:
        gpu_dfs = []
        
        for result_dir in result_dirs:
            result_data = load_result_data(result_dir)
            if "gpu_metrics" in result_data:
                gpu_df = preprocess_metrics_data(result_data["gpu_metrics"])
                gpu_dfs.append(gpu_df)
            else:
                gpu_dfs.append(pd.DataFrame())
        
        if any(not df.empty for df in gpu_dfs):
            output_file = os.path.join(output_dir, "gpu_comparison.png")
            title = "GPU 메트릭 비교"
            if compare_gpu_metrics(gpu_dfs, names, title, output_file):
                print(f"GPU 메트릭 비교 차트가 생성되었습니다: {output_file}")
                success = True
    
    return success

def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description="GenAI 성능 테스트 결과 시각화")
    
    # 결과 디렉토리 인수 그룹
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result_dir", help="단일 결과 디렉토리 경로")
    group.add_argument("--compare", nargs="+", help="비교할 여러 결과 디렉토리 경로")
    group.add_argument("--latest", action="store_true", help="최신 결과 시각화")
    group.add_argument("--model", help="특정 모델의 최신 결과 시각화")
    
    # 추가 옵션
    parser.add_argument("--output_dir", help="출력 디렉토리 경로")
    parser.add_argument("--chart_types", nargs="+", choices=["latency", "throughput", "percentile", "gpu", "performance"], 
                        help="생성할 차트 유형")
    parser.add_argument("--names", nargs="+", help="비교 시 사용할 이름 (--compare와 함께 사용)")
    parser.add_argument("--artifacts_dir", default="/artifacts", help="아티팩트 디렉토리 경로")
    
    args = parser.parse_args()
    
    # 단일 결과 시각화
    if args.result_dir:
        return visualize_single_result(args.result_dir, args.output_dir, args.chart_types)
    
    # 여러 결과 비교 시각화
    elif args.compare:
        return compare_multiple_results(args.compare, args.output_dir, args.chart_types, args.names)
    
    # 최신 결과 시각화
    elif args.latest or args.model:
        result_dirs = find_result_dirs(args.artifacts_dir)
        
        if not result_dirs:
            print(f"오류: {args.artifacts_dir}에서 결과 디렉토리를 찾을 수 없습니다.")
            return False
        
        if args.model:
            # 특정 모델의 최신 결과 찾기
            model_dirs = [d for d in result_dirs if args.model.lower() in get_model_name(d).lower()]
            if not model_dirs:
                print(f"오류: 모델 '{args.model}'의 결과를 찾을 수 없습니다.")
                return False
            
            latest_dir = model_dirs[0]
        else:
            # 가장 최신 결과 사용
            latest_dir = result_dirs[0]
        
        print(f"최신 결과 디렉토리: {latest_dir}")
        return visualize_single_result(latest_dir, args.output_dir, args.chart_types)
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
