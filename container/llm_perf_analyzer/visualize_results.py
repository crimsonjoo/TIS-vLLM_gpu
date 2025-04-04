#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import argparse
from io import StringIO
from datetime import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 시각화를 위한 스타일 설정
plt.style.use('ggplot')
COLORS = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8c6bb1', '#fdb462', '#fb8072', '#80b1d3']
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def parse_csv(csv_file):
    """CSV 파일을 파싱하여 데이터프레임으로 반환합니다."""
    # CSV 파일 읽기
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # 섹션 분리
    sections = []
    current_section = []
    
    for line in lines:
        if line.strip() == '':
            if current_section:
                sections.append(current_section)
                current_section = []
        else:
            current_section.append(line.strip())
    
    if current_section:
        sections.append(current_section)
    
    # 각 섹션을 데이터프레임으로 변환
    dfs = []
    for section in sections:
        if not section:
            continue
        
        # CSV 문자열 생성
        csv_str = '\n'.join(section)
        
        # 데이터프레임으로 변환
        df = pd.read_csv(StringIO(csv_str))
        dfs.append(df)
    
    return dfs

def clean_value(value):
    """문자열 값을 숫자로 변환합니다."""
    if isinstance(value, str) and ',' in value:
        return float(value.replace(',', ''))
    return value

def create_latency_insights_chart(df, title, output_file):
    """지연 시간 인사이트 차트를 생성합니다."""
    if 'Metric' not in df.columns:
        return False
    
    # 지연 시간 관련 지표 필터링
    latency_metrics = [
        'Time To First Token (ms)', 
        'Time To Second Token (ms)', 
        'Request Latency (ms)', 
        'Inter Token Latency (ms)'
    ]
    
    latency_df = df[df['Metric'].isin(latency_metrics)]
    
    if len(latency_df) == 0:
        return False
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. 지연 시간 비교 차트 (왼쪽 상단)
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = []
    avg_values = []
    p99_values = []
    
    for i, row in latency_df.iterrows():
        metric = row['Metric']
        if metric == 'Request Latency (ms)':
            # Request Latency는 너무 크므로 별도 축으로 처리
            continue
        
        metrics.append(metric.replace(' (ms)', ''))
        avg_values.append(clean_value(row['avg']))
        p99_values.append(clean_value(row['p99']))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, avg_values, width, label='평균', color=COLORS[0])
    ax1.bar(x + width/2, p99_values, width, label='p99', color=COLORS[1])
    
    ax1.set_ylabel('지연 시간 (ms)')
    ax1.set_title('지연 시간 지표 비교 (Request Latency 제외)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    
    # 2. 지연 시간 일관성 차트 (오른쪽 상단)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, metric in enumerate(metrics):
        percentiles = ['min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max']
        values = []
        
        for p in percentiles:
            values.append(clean_value(latency_df.iloc[i][p]))
        
        ax2.plot(percentiles, values, marker='o', label=metric, color=COLORS[i])
    
    ax2.set_ylabel('지연 시간 (ms)')
    ax2.set_title('지연 시간 백분위수 분포')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. TTFT vs Inter Token Latency 비교 (왼쪽 하단)
    ax3 = fig.add_subplot(gs[1, 0])
    
    ttft_row = latency_df[latency_df['Metric'] == 'Time To First Token (ms)']
    itl_row = latency_df[latency_df['Metric'] == 'Inter Token Latency (ms)']
    
    if not ttft_row.empty and not itl_row.empty:
        ttft_avg = clean_value(ttft_row.iloc[0]['avg'])
        itl_avg = clean_value(itl_row.iloc[0]['avg'])
        
        labels = ['TTFT', 'Inter Token Latency']
        values = [ttft_avg, itl_avg]
        
        ax3.bar(labels, values, color=[COLORS[0], COLORS[2]])
        ax3.set_ylabel('지연 시간 (ms)')
        ax3.set_title('TTFT vs Inter Token Latency')
        
        # TTFT/ITL 비율 표시
        ratio = ttft_avg / itl_avg
        ax3.text(0, ttft_avg * 1.05, f'비율: {ratio:.1f}x', ha='center')
        
        # 해석 추가
        if ratio > 5:
            interpretation = "TTFT가 매우 높음: 모델 초기화 최적화 필요"
        elif ratio > 3:
            interpretation = "TTFT가 높음: 컨텍스트 처리 검토 필요"
        else:
            interpretation = "TTFT/ITL 비율 양호"
        
        ax3.text(0.5, max(values) * 0.5, interpretation, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # 4. 요청 지연 시간 분석 (오른쪽 하단)
    ax4 = fig.add_subplot(gs[1, 1])
    
    req_latency_row = latency_df[latency_df['Metric'] == 'Request Latency (ms)']
    
    if not req_latency_row.empty and not ttft_row.empty and not itl_row.empty:
        req_latency_avg = clean_value(req_latency_row.iloc[0]['avg'])
        ttft_avg = clean_value(ttft_row.iloc[0]['avg'])
        itl_avg = clean_value(itl_row.iloc[0]['avg'])
        
        # 출력 토큰 수 가져오기
        output_length_df = df[df['Metric'] == 'Output Sequence Length (tokens)']
        if not output_length_df.empty:
            output_tokens = clean_value(output_length_df.iloc[0]['avg'])
        else:
            # 출력 토큰 수 정보가 없는 경우 추정
            output_tokens = (req_latency_avg - ttft_avg) / itl_avg
        
        # 요청 지연 시간 분석
        ttft_portion = ttft_avg
        generation_portion = itl_avg * (output_tokens - 1)  # 첫 번째 토큰 제외
        other_portion = req_latency_avg - ttft_portion - generation_portion
        
        labels = ['TTFT', '토큰 생성', '기타']
        sizes = [ttft_portion, generation_portion, other_portion]
        
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=[COLORS[0], COLORS[2], COLORS[3]])
        ax4.set_title('요청 지연 시간 구성')
        
        # 해석 추가
        ttft_pct = ttft_portion / req_latency_avg * 100
        gen_pct = generation_portion / req_latency_avg * 100
        
        if ttft_pct > 50:
            interpretation = "TTFT가 병목: 초기화 최적화 필요"
        elif gen_pct > 70:
            interpretation = "토큰 생성이 대부분: 생성 속도 최적화 필요"
        else:
            interpretation = "균형 잡힌 지연 시간 구성"
        
        ax4.text(0, -1.2, interpretation, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_throughput_insights_chart(df, title, output_file):
    """처리량 인사이트 차트를 생성합니다."""
    throughput_metrics = ['Output Token Throughput (per sec)', 'Request Throughput (per sec)']
    
    throughput_df = None
    for i, d in enumerate(df):
        if 'Metric' in d.columns and 'Value' in d.columns:
            filtered = d[d['Metric'].isin(throughput_metrics)]
            if not filtered.empty:
                throughput_df = filtered
                break
    
    if throughput_df is None or len(throughput_df) == 0:
        return False
    
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    
    # 1. 처리량 차트 (왼쪽)
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = []
    values = []
    
    for i, row in throughput_df.iterrows():
        metric = row['Metric'].replace(' (per sec)', '')
        metrics.append(metric)
        values.append(clean_value(row['Value']))
    
    ax1.bar(metrics, values, color=[COLORS[0], COLORS[1]])
    
    for i, v in enumerate(values):
        ax1.text(i, v * 1.05, f'{v:.2f}', ha='center')
    
    ax1.set_ylabel('처리량 (단위/초)')
    ax1.set_title('토큰 및 요청 처리량')
    
    # 2. 처리량 해석 (오른쪽)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # 출력 토큰 처리량 해석
    token_throughput = None
    req_throughput = None
    
    for i, row in throughput_df.iterrows():
        if row['Metric'] == 'Output Token Throughput (per sec)':
            token_throughput = clean_value(row['Value'])
        elif row['Metric'] == 'Request Throughput (per sec)':
            req_throughput = clean_value(row['Value'])
    
    if token_throughput is not None:
        interpretation = ""
        
        if token_throughput < 20:
            interpretation += "• 토큰 처리량이 매우 낮음 (< 20 tokens/sec)\n"
            interpretation += "  - 가능한 원인: 모델 크기가 매우 큼, 하드웨어 제약, 비효율적인 서빙 설정\n"
            interpretation += "  - 권장 조치: 모델 양자화 검토, 배치 처리 최적화, 하드웨어 업그레이드 고려\n\n"
        elif token_throughput < 40:
            interpretation += "• 토큰 처리량이 낮음 (20-40 tokens/sec)\n"
            interpretation += "  - 가능한 원인: 큰 모델 크기, 서빙 구성 최적화 필요\n"
            interpretation += "  - 권장 조치: KV 캐시 최적화, 배치 크기 조정, 모델 최적화 검토\n\n"
        elif token_throughput < 80:
            interpretation += "• 토큰 처리량이 보통 (40-80 tokens/sec)\n"
            interpretation += "  - 현재 상태: 일반적인 대형 LLM 모델의 성능 범위 내\n"
            interpretation += "  - 추가 최적화: 필요한 경우 추론 엔진 튜닝 검토\n\n"
        else:
            interpretation += "• 토큰 처리량이 높음 (> 80 tokens/sec)\n"
            interpretation += "  - 현재 상태: 매우 효율적인 모델 서빙 구성\n"
            interpretation += "  - 참고: 처리량과 지연 시간의 균형 확인\n\n"
    
        if req_throughput is not None:
            interpretation += f"• 요청 처리량: {req_throughput:.2f} req/sec\n"
            interpretation += f"  - 초당 약 {int(req_throughput * 60)} 요청/분 처리 가능\n"
            
            # 동시성 처리량 추정
            interpretation += f"  - 평균 요청 처리 시간 기준 최대 동시성: {int(req_throughput * 5) + 1} 요청\n\n"
        
        # 실용적 해석
        interpretation += "• 실제 사용 시나리오 추정:\n"
        if token_throughput < 30:
            interpretation += "  - 소규모 사용자 기반에 적합 (< 10명 동시 사용자)\n"
            interpretation += "  - 대화형 응용 프로그램에 제한적 사용 권장\n"
        elif token_throughput < 60:
            interpretation += "  - 중간 규모 사용자 기반에 적합 (10-30명 동시 사용자)\n"
            interpretation += "  - 대화형 응용 프로그램에 적합\n"
        else:
            interpretation += "  - 대규모 사용자 기반에 적합 (30+ 동시 사용자)\n"
            interpretation += "  - 높은 처리량이 필요한 응용 프로그램에 적합\n"
        
        ax2.text(0, 0.95, "처리량 분석 및 인사이트", fontsize=14, weight='bold')
        ax2.text(0, 0.85, interpretation, fontsize=11, va='top', linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_bar_chart(df, title, output_file):
    """막대 그래프를 생성합니다."""
    plt.figure(figsize=(12, 6))
    
    # 첫 번째 열이 메트릭 이름인 경우
    if 'Metric' in df.columns:
        metrics = df['Metric'].tolist()
        
        # 'avg' 열이 있는 경우
        if 'avg' in df.columns:
            values = df['avg'].tolist()
            plt.bar(metrics, values)
            plt.title(title)
            plt.ylabel('Average Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            return True
        
        # 'Value' 열이 있는 경우
        elif 'Value' in df.columns:
            values = df['Value'].tolist()
            plt.bar(metrics, values)
            plt.title(title)
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            return True
    
    return False

def create_percentile_chart(df, title, output_file):
    """백분위수 차트를 생성합니다."""
    if 'Metric' not in df.columns:
        return False
    
    percentile_cols = ['avg', 'min', 'max', 'p99', 'p90', 'p75', 'p50']
    available_cols = [col for col in percentile_cols if col in df.columns]
    
    if not available_cols:
        return False
    
    metrics = df['Metric'].tolist()
    
    plt.figure(figsize=(15, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        
        values = [df.loc[i, col] for col in available_cols]
        
        # 문자열 값을 숫자로 변환 (쉼표 제거)
        numeric_values = []
        for val in values:
            if isinstance(val, str) and ',' in val:
                val = float(val.replace(',', ''))
            numeric_values.append(val)
        
        plt.bar(available_cols, numeric_values)
        plt.title(f'{metric}')
        plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    return True

def create_gpu_metrics_chart(df, title, output_file):
    """GPU 메트릭 차트를 생성합니다."""
    if 'GPU' not in df.columns or 'Metric' not in df.columns:
        return False
    
    # GPU별로 그룹화
    gpus = df['GPU'].unique()
    metrics = df['Metric'].unique()
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        
        for gpu in gpus:
            gpu_data = df[(df['GPU'] == gpu) & (df['Metric'] == metric)]
            
            if 'avg' in df.columns:
                plt.bar(f"{gpu} - {metric}", gpu_data['avg'].values[0], label=gpu)
            elif 'Value' in df.columns:
                plt.bar(f"{gpu} - {metric}", gpu_data['Value'].values[0], label=gpu)
        
        plt.title(f'{metric}')
        plt.legend()
        plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    return True

def create_comparison_chart(results_dir, output_dir):
    """여러 테스트 결과를 비교하는 차트를 생성합니다."""
    # 결과 디렉토리 목록 가져오기
    result_dirs = [d for d in os.listdir(results_dir) if d.startswith('results_')]
    
    if not result_dirs:
        print("비교할 결과가 없습니다.")
        return
    
    # 각 결과 디렉토리에서 데이터 수집
    comparison_data = {}
    
    for result_dir in result_dirs:
        # 파라미터 파일 읽기
        params_file = os.path.join(results_dir, result_dir, 'parameters.txt')
        if not os.path.exists(params_file):
            continue
        
        with open(params_file, 'r') as f:
            params = f.readlines()
        
        # 모델 이름 및 백엔드 추출
        model_name = None
        backend = None
        
        for line in params:
            if line.startswith('모델 이름:'):
                model_name = line.split(':')[1].strip()
            elif line.startswith('백엔드:'):
                backend = line.split(':')[1].strip()
        
        if not model_name or not backend:
            continue
        
        # 결과 CSV 파일 찾기
        csv_files = []
        for root, _, files in os.walk(os.path.join(results_dir, result_dir)):
            for file in files:
                if file.endswith('_genai_perf.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            continue
        
        # 첫 번째 CSV 파일 사용
        csv_file = csv_files[0]
        dfs = parse_csv(csv_file)
        
        if not dfs:
            continue
        
        # 주요 메트릭 추출
        for df in dfs:
            if 'Metric' in df.columns and ('avg' in df.columns or 'Value' in df.columns):
                for i, row in df.iterrows():
                    metric = row['Metric']
                    
                    if 'avg' in df.columns:
                        value = row['avg']
                    elif 'Value' in df.columns:
                        value = row['Value']
                    else:
                        continue
                    
                    # 문자열 값을 숫자로 변환 (쉼표 제거)
                    if isinstance(value, str) and ',' in value:
                        value = float(value.replace(',', ''))
                    
                    if metric not in comparison_data:
                        comparison_data[metric] = {}
                    
                    comparison_data[metric][f"{model_name}-{backend}"] = value
    
    # 비교 차트 생성
    for metric, values in comparison_data.items():
        plt.figure(figsize=(12, 6))
        
        models = list(values.keys())
        metric_values = list(values.values())
        
        plt.bar(models, metric_values)
        plt.title(f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 출력 파일 이름 생성
        output_file = os.path.join(output_dir, f'comparison_{metric.replace(" ", "_").lower()}.png')
        plt.savefig(output_file)
        plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='GenAI-Perf 결과 시각화')
    parser.add_argument('--result-dir', type=str, required=True, help='결과 디렉토리 경로')
    parser.add_argument('--output-dir', type=str, default=None, help='출력 디렉토리 경로 (기본값: 결과 디렉토리 내 visualizations)')
    parser.add_argument('--compare', action='store_true', help='여러 테스트 결과 비교')
    
    args = parser.parse_args()
    
    if args.compare:
        # 여러 테스트 결과 비교
        results_dir = os.path.dirname(args.result_dir)
        output_dir = args.output_dir or os.path.join(results_dir, 'comparisons')
        os.makedirs(output_dir, exist_ok=True)
        
        create_comparison_chart(results_dir, output_dir)
        print(f"비교 차트가 {output_dir}에 생성되었습니다.")
        return
    
    # 단일 테스트 결과 시각화
    result_dir = args.result_dir
    output_dir = args.output_dir or os.path.join(result_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일 찾기
    csv_files = []
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith('_genai_perf.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"결과 디렉토리 {result_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return
    
    # 각 CSV 파일 처리
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # 파일 이름에서 모델 및 백엔드 정보 추출
        filename = os.path.basename(csv_file)
        model_info = filename.replace('_genai_perf.csv', '')
        
        # CSV 파일 파싱
        dfs = parse_csv(csv_file)
        
        # 각 데이터프레임에 대해 차트 생성
        for i, df in enumerate(dfs):
            # 기본 차트 이름
            chart_name = f"{model_info}_section{i}"
            
            # 섹션 유형 확인
            if 'Metric' in df.columns and 'GPU' in df.columns:
                # GPU 메트릭 차트
                output_file = os.path.join(output_dir, f"{chart_name}_gpu_metrics.png")
                create_gpu_metrics_chart(df, f"GPU Metrics - {model_info}", output_file)
            elif 'Metric' in df.columns and len(df) <= 3:
                # 간단한 메트릭 차트 (처리량 등)
                output_file = os.path.join(output_dir, f"{chart_name}_simple_metrics.png")
                create_bar_chart(df, f"Performance Metrics - {model_info}", output_file)
            elif 'Metric' in df.columns:
                # 백분위수 차트
                output_file = os.path.join(output_dir, f"{chart_name}_percentile.png")
                create_percentile_chart(df, f"Percentile Metrics - {model_info}", output_file)
            
            # 지연 시간 인사이트 차트
            output_file = os.path.join(output_dir, f"{chart_name}_latency_insights.png")
            create_latency_insights_chart(df, f"Latency Insights - {model_info}", output_file)
            
            # 처리량 인사이트 차트
            output_file = os.path.join(output_dir, f"{chart_name}_throughput_insights.png")
            create_throughput_insights_chart(df, f"Throughput Insights - {model_info}", output_file)
    
    print(f"시각화가 {output_dir}에 생성되었습니다.")

if __name__ == "__main__":
    main()
