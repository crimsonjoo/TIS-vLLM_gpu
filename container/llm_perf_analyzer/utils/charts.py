#!/usr/bin/env python3
"""
GenAI 성능 테스트 결과 차트 생성 모듈

이 모듈은 GenAI-Perf 성능 테스트 결과를 시각화하는 차트 생성 함수들을 제공합니다.
다양한 성능 지표에 대한 차트를 생성하는 기능을 담당합니다.

주요 기능:
- 지연 시간 인사이트 차트 생성
- 처리량 인사이트 차트 생성
- GPU 메트릭 차트 생성
- 백분위수 차트 생성
- 여러 테스트 결과 비교 차트 생성
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
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

def clean_value(value):
    """
    문자열 값을 숫자로 변환합니다.
    
    Args:
        value: 변환할 값
        
    Returns:
        float: 변환된 숫자 값
    """
    if isinstance(value, str) and ',' in value:
        return float(value.replace(',', ''))
    return value

def create_latency_insights_chart(df, title, output_file):
    """
    지연 시간 인사이트 차트를 생성합니다.
    
    Args:
        df (DataFrame): 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
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
            interpretation = "TTFT가 지연 시간의 주요 원인"
        elif gen_pct > 70:
            interpretation = "토큰 생성이 지연 시간의 주요 원인"
        else:
            interpretation = "지연 시간 구성 균형적"
        
        plt.figtext(0.5, 0.02, interpretation, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_throughput_insights_chart(df, title, output_file):
    """
    처리량 인사이트 차트를 생성합니다.
    
    Args:
        df (DataFrame): 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if 'Metric' not in df.columns:
        return False
    
    # 처리량 관련 지표 필터링
    throughput_metrics = [
        'Output Token Throughput (tokens/sec)', 
        'Request Throughput (req/sec)',
        'Output Sequence Length (tokens)',
        'Input Sequence Length (tokens)'
    ]
    
    throughput_df = df[df['Metric'].isin(throughput_metrics)]
    
    if len(throughput_df) == 0:
        return False
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. 토큰 처리량 차트 (왼쪽 상단)
    ax1 = fig.add_subplot(gs[0, 0])
    
    token_throughput_row = throughput_df[throughput_df['Metric'] == 'Output Token Throughput (tokens/sec)']
    if not token_throughput_row.empty:
        token_throughput = clean_value(token_throughput_row.iloc[0]['avg'])
        
        ax1.bar(['토큰 처리량'], [token_throughput], color=COLORS[0])
        ax1.set_ylabel('토큰/초')
        ax1.set_title('출력 토큰 처리량')
        
        # 처리량 해석
        if token_throughput > 100:
            interpretation = "높은 처리량: 매우 효율적"
        elif token_throughput > 50:
            interpretation = "양호한 처리량"
        else:
            interpretation = "낮은 처리량: 최적화 필요"
        
        ax1.text(0, token_throughput * 0.5, interpretation, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # 2. 요청 처리량 차트 (오른쪽 상단)
    ax2 = fig.add_subplot(gs[0, 1])
    
    req_throughput_row = throughput_df[throughput_df['Metric'] == 'Request Throughput (req/sec)']
    if not req_throughput_row.empty:
        req_throughput = clean_value(req_throughput_row.iloc[0]['avg'])
        
        ax2.bar(['요청 처리량'], [req_throughput], color=COLORS[1])
        ax2.set_ylabel('요청/초')
        ax2.set_title('요청 처리량')
        
        # 처리량 해석
        if req_throughput > 10:
            interpretation = "높은 요청 처리량"
        elif req_throughput > 1:
            interpretation = "양호한 요청 처리량"
        else:
            interpretation = "낮은 요청 처리량"
        
        ax2.text(0, req_throughput * 0.5, interpretation, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # 3. 시퀀스 길이 차트 (왼쪽 하단)
    ax3 = fig.add_subplot(gs[1, 0])
    
    input_len_row = throughput_df[throughput_df['Metric'] == 'Input Sequence Length (tokens)']
    output_len_row = throughput_df[throughput_df['Metric'] == 'Output Sequence Length (tokens)']
    
    if not input_len_row.empty and not output_len_row.empty:
        input_len = clean_value(input_len_row.iloc[0]['avg'])
        output_len = clean_value(output_len_row.iloc[0]['avg'])
        
        ax3.bar(['입력 시퀀스', '출력 시퀀스'], [input_len, output_len], color=[COLORS[2], COLORS[3]])
        ax3.set_ylabel('토큰 수')
        ax3.set_title('평균 시퀀스 길이')
        
        # 비율 표시
        ratio = output_len / input_len if input_len > 0 else 0
        ax3.text(0, input_len * 1.05, f'입력', ha='center')
        ax3.text(1, output_len * 1.05, f'출력 (비율: {ratio:.1f}x)', ha='center')
    
    # 4. 토큰 생성 효율성 차트 (오른쪽 하단)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if (not token_throughput_row.empty and not req_throughput_row.empty and 
        not input_len_row.empty and not output_len_row.empty):
        
        token_throughput = clean_value(token_throughput_row.iloc[0]['avg'])
        req_throughput = clean_value(req_throughput_row.iloc[0]['avg'])
        input_len = clean_value(input_len_row.iloc[0]['avg'])
        output_len = clean_value(output_len_row.iloc[0]['avg'])
        
        # 효율성 지표 계산
        tokens_per_request = output_len
        processing_time = 1 / req_throughput if req_throughput > 0 else 0
        token_generation_rate = token_throughput / req_throughput if req_throughput > 0 else 0
        
        metrics = ['토큰/요청', '처리 시간(초)', '생성 속도(토큰/요청/초)']
        values = [tokens_per_request, processing_time, token_generation_rate]
        
        ax4.bar(metrics, values, color=[COLORS[4], COLORS[5], COLORS[6]])
        ax4.set_ylabel('값')
        ax4.set_title('토큰 생성 효율성')
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_bar_chart(df, title, output_file):
    """
    막대 그래프를 생성합니다.
    
    Args:
        df (DataFrame): 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if 'Metric' not in df.columns or 'avg' not in df.columns:
        return False
    
    plt.figure(figsize=(10, 6))
    
    metrics = df['Metric'].tolist()
    values = [clean_value(val) for val in df['avg'].tolist()]
    
    plt.bar(metrics, values, color=COLORS[:len(metrics)])
    plt.ylabel('값')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_percentile_chart(df, title, output_file):
    """
    백분위수 차트를 생성합니다.
    
    Args:
        df (DataFrame): 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if 'Metric' not in df.columns:
        return False
    
    percentile_cols = ['min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max']
    if not all(col in df.columns for col in percentile_cols):
        return False
    
    plt.figure(figsize=(12, 8))
    
    for i, row in df.iterrows():
        metric = row['Metric']
        values = [clean_value(row[col]) for col in percentile_cols]
        
        plt.plot(percentile_cols, values, marker='o', label=metric, color=COLORS[i % len(COLORS)])
    
    plt.ylabel('값')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_gpu_metrics_chart(df, title, output_file):
    """
    GPU 메트릭 차트를 생성합니다.
    
    Args:
        df (DataFrame): 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if 'Metric' not in df.columns or 'GPU' not in df.columns or 'avg' not in df.columns:
        return False
    
    plt.figure(figsize=(12, 8))
    
    # GPU별로 그룹화
    gpu_groups = df.groupby('GPU')
    
    for i, (gpu, group) in enumerate(gpu_groups):
        metrics = group['Metric'].tolist()
        values = [clean_value(val) for val in group['avg'].tolist()]
        
        x = np.arange(len(metrics))
        plt.bar(x + i*0.25, values, width=0.25, label=f'GPU {gpu}', color=COLORS[i % len(COLORS)])
    
    plt.ylabel('값')
    plt.title(title)
    plt.xticks(np.arange(len(metrics)) + 0.25*(len(gpu_groups)-1)/2, metrics, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    
    return True
