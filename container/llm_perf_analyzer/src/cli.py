#!/usr/bin/env python3
"""
GenAI 성능 분석기 CLI 모듈

이 모듈은 GenAI 성능 분석기의 명령줄 인터페이스를 제공합니다.
성능 테스트 결과 시각화 및 분석을 위한 다양한 명령어를 지원합니다.

주요 기능:
- 성능 테스트 실행 명령어
- 결과 시각화 명령어
- 결과 비교 분석 명령어
"""

import sys
import argparse
from pathlib import Path

# 상위 디렉토리를 시스템 경로에 추가하여 모듈에 접근할 수 있도록 함
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    visualize_single_result,
    compare_multiple_results,
    find_result_dirs
)

def parse_args():
    """
    명령줄 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 명령줄 인수
    """
    parser = argparse.ArgumentParser(description="GenAI 성능 테스트 결과 시각화")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result_dir", help="단일 결과 디렉토리 경로")
    group.add_argument("--compare", nargs="+", help="비교할 여러 결과 디렉토리 경로")
    group.add_argument("--latest", action="store_true", help="최신 결과 시각화")
    group.add_argument("--model", help="특정 모델의 최신 결과 시각화")
    
    parser.add_argument("--output_dir", help="출력 디렉토리 경로")
    parser.add_argument("--chart_types", nargs="+", 
                        choices=["latency", "throughput", "percentile", "gpu", "performance"],
                        help="생성할 차트 유형")
    parser.add_argument("--names", nargs="+", help="비교 시 사용할 이름 (--compare와 함께 사용)")
    parser.add_argument("--artifacts_dir", help="아티팩트 디렉토리 경로")
    
    return parser.parse_args()

def run_visualization():
    """
    명령줄 인수에 따라 시각화 기능을 실행합니다.
    
    Returns:
        bool: 성공 여부
    """
    args = parse_args()
    
    # 아티팩트 디렉토리 설정
    artifacts_dir = args.artifacts_dir or "/workspace/llm_perf_analyzer/artifacts"
    
    if args.result_dir:
        # 단일 결과 시각화
        return visualize_single_result(
            args.result_dir, 
            args.output_dir, 
            args.chart_types
        )
    elif args.compare:
        # 여러 결과 비교 시각화
        return compare_multiple_results(
            args.compare, 
            args.output_dir, 
            args.chart_types, 
            args.names
        )
    elif args.latest:
        # 최신 결과 시각화
        result_dirs = find_result_dirs(artifacts_dir)
        if not result_dirs:
            print(f"오류: {artifacts_dir} 디렉토리에 결과가 없습니다.")
            return False
        latest_dir = result_dirs[-1]
        print(f"최신 결과 디렉토리: {latest_dir}")
        return visualize_single_result(
            latest_dir, 
            args.output_dir, 
            args.chart_types
        )
    elif args.model:
        # 특정 모델의 최신 결과 시각화
        result_dirs = find_result_dirs(artifacts_dir)
        if not result_dirs:
            print(f"오류: {artifacts_dir} 디렉토리에 결과가 없습니다.")
            return False
        
        # 모델 이름으로 필터링
        model_dirs = [d for d in result_dirs if args.model in Path(d).name]
        if not model_dirs:
            print(f"오류: {args.model} 모델의 결과가 없습니다.")
            return False
        
        latest_dir = model_dirs[-1]
        print(f"{args.model} 모델의 최신 결과 디렉토리: {latest_dir}")
        return visualize_single_result(
            latest_dir, 
            args.output_dir, 
            args.chart_types
        )
    
    return False

def main():
    """
    메인 함수
    
    Returns:
        bool: 성공 여부
    """
    try:
        return run_visualization()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
