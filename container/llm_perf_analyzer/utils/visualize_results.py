#!/usr/bin/env python3
"""
GenAI 성능 테스트 결과 시각화 모듈

이 모듈은 GenAI-Perf 성능 테스트 결과를 시각화하는 기능을 제공합니다.
다양한 성능 지표에 대한 차트를 생성하고, 결과를 분석하는 데 도움이 됩니다.

주요 기능:
- CSV 파일 파싱 및 데이터 처리
- 지연 시간 인사이트 차트 생성
- 처리량 인사이트 차트 생성
- GPU 메트릭 차트 생성
- 백분위수 차트 생성
- 여러 테스트 결과 비교 차트 생성
"""

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
    """
    CSV 파일을 파싱하여 데이터프레임으로 반환합니다.
    
    Args:
        csv_file (str): CSV 파일 경로
        
    Returns:
        list: 데이터프레임 리스트
    """
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
