#!/usr/bin/env python3
"""
GenAI 성능 테스트 결과 비교 모듈

이 모듈은 여러 GenAI-Perf 성능 테스트 결과를 비교하는 함수들을 제공합니다.
서로 다른 모델이나 설정에 대한 성능 테스트 결과를 비교하여 시각화합니다.

주요 기능:
- 여러 테스트 결과의 지연 시간 비교
- 여러 테스트 결과의 처리량 비교
- 여러 테스트 결과의 GPU 사용률 비교
- 여러 테스트 결과의 백분위수 비교
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

def compare_results(result_dfs, names, title, output_file, metrics=None):
    """
    여러 테스트 결과를 비교하는 차트를 생성합니다.
    
    Args:
        result_dfs (list): 데이터프레임 리스트
        names (list): 각 결과의 이름 리스트
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        metrics (list, optional): 비교할 지표 리스트. 기본값은 None.
        
    Returns:
        bool: 성공 여부
    """
    if not result_dfs or len(result_dfs) != len(names):
        return False
    
    # 기본 비교 지표 설정
    if metrics is None:
        metrics = [
            'Request Latency (ms)',
            'Time To First Token (ms)',
            'Inter Token Latency (ms)',
            'Output Token Throughput (tokens/sec)',
            'Request Throughput (req/sec)'
        ]
    
    # 각 결과에서 지표 추출
    comparison_data = {}
    for metric in metrics:
        comparison_data[metric] = []
        
        for df in result_dfs:
            if 'Metric' not in df.columns or 'avg' not in df.columns:
                comparison_data[metric].append(0)
                continue
                
            metric_row = df[df['Metric'] == metric]
            if not metric_row.empty:
                comparison_data[metric].append(clean_value(metric_row.iloc[0]['avg']))
            else:
                comparison_data[metric].append(0)
    
    # 차트 생성
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. 지연 시간 비교 차트 (왼쪽 상단)
    ax1 = fig.add_subplot(gs[0, 0])
    
    latency_metrics = [m for m in metrics if 'Latency' in m or 'Token' in m and 'Throughput' not in m]
    if latency_metrics:
        x = np.arange(len(names))
        width = 0.8 / len(latency_metrics)
        
        for i, metric in enumerate(latency_metrics):
            values = comparison_data[metric]
            ax1.bar(x + i*width - 0.4 + width/2, values, width, label=metric, color=COLORS[i % len(COLORS)])
        
        ax1.set_ylabel('지연 시간 (ms)')
        ax1.set_title('지연 시간 지표 비교')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
    
    # 2. 처리량 비교 차트 (오른쪽 상단)
    ax2 = fig.add_subplot(gs[0, 1])
    
    throughput_metrics = [m for m in metrics if 'Throughput' in m]
    if throughput_metrics:
        x = np.arange(len(names))
        width = 0.8 / len(throughput_metrics)
        
        for i, metric in enumerate(throughput_metrics):
            values = comparison_data[metric]
            ax2.bar(x + i*width - 0.4 + width/2, values, width, label=metric, color=COLORS[i % len(COLORS)])
        
        ax2.set_ylabel('처리량')
        ax2.set_title('처리량 지표 비교')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.legend()
    
    # 3. 지연 시간 백분위수 비교 (왼쪽 하단)
    ax3 = fig.add_subplot(gs[1, 0])
    
    if 'Request Latency (ms)' in metrics:
        percentile_cols = ['min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max']
        
        for i, df in enumerate(result_dfs):
            if 'Metric' not in df.columns or not all(col in df.columns for col in percentile_cols):
                continue
                
            latency_row = df[df['Metric'] == 'Request Latency (ms)']
            if not latency_row.empty:
                values = [clean_value(latency_row.iloc[0][col]) for col in percentile_cols]
                ax3.plot(percentile_cols, values, marker='o', label=names[i], color=COLORS[i % len(COLORS)])
        
        ax3.set_ylabel('지연 시간 (ms)')
        ax3.set_title('요청 지연 시간 백분위수 비교')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 성능 개선율 차트 (오른쪽 하단)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if len(result_dfs) > 1:
        # 첫 번째 결과를 기준으로 개선율 계산
        improvement_metrics = []
        improvement_values = []
        
        for metric in metrics:
            if comparison_data[metric][0] == 0:
                continue
                
            for i in range(1, len(result_dfs)):
                if comparison_data[metric][i] == 0:
                    continue
                    
                # 지연 시간은 낮을수록 좋고, 처리량은 높을수록 좋음
                if 'Latency' in metric or 'Time' in metric:
                    improvement = (comparison_data[metric][0] - comparison_data[metric][i]) / comparison_data[metric][0] * 100
                else:
                    improvement = (comparison_data[metric][i] - comparison_data[metric][0]) / comparison_data[metric][0] * 100
                
                improvement_metrics.append(f"{metric} ({names[i]} vs {names[0]})")
                improvement_values.append(improvement)
        
        if improvement_metrics:
            # 개선율 순으로 정렬
            sorted_indices = np.argsort(improvement_values)
            sorted_metrics = [improvement_metrics[i] for i in sorted_indices]
            sorted_values = [improvement_values[i] for i in sorted_indices]
            
            colors = ['#EA4335' if v < 0 else '#34A853' for v in sorted_values]
            ax4.barh(sorted_metrics, sorted_values, color=colors)
            ax4.set_xlabel('개선율 (%)')
            ax4.set_title('성능 개선율 (기준: ' + names[0] + ')')
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 값 표시
            for i, v in enumerate(sorted_values):
                ax4.text(v + np.sign(v) * 1, i, f"{v:.1f}%", va='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True

def compare_gpu_metrics(result_dfs, names, title, output_file):
    """
    여러 테스트 결과의 GPU 메트릭을 비교하는 차트를 생성합니다.
    
    Args:
        result_dfs (list): 데이터프레임 리스트
        names (list): 각 결과의 이름 리스트
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if not result_dfs or len(result_dfs) != len(names):
        return False
    
    # GPU 메트릭 필터링
    gpu_metrics = [
        'GPU Utilization (%)',
        'GPU Memory Used (MB)',
        'GPU Power (W)'
    ]
    
    # 각 결과에서 GPU 메트릭 추출
    has_gpu_data = False
    comparison_data = {}
    
    for metric in gpu_metrics:
        comparison_data[metric] = []
        
        for df in result_dfs:
            if 'Metric' not in df.columns or 'GPU' not in df.columns or 'avg' not in df.columns:
                comparison_data[metric].append([])
                continue
                
            metric_rows = df[df['Metric'] == metric]
            if not metric_rows.empty:
                has_gpu_data = True
                gpu_values = []
                
                for gpu in sorted(metric_rows['GPU'].unique()):
                    gpu_row = metric_rows[metric_rows['GPU'] == gpu]
                    if not gpu_row.empty:
                        gpu_values.append((gpu, clean_value(gpu_row.iloc[0]['avg'])))
                
                comparison_data[metric].append(gpu_values)
            else:
                comparison_data[metric].append([])
    
    if not has_gpu_data:
        return False
    
    # 차트 생성
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. GPU 사용률 비교 차트 (왼쪽 상단)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if comparison_data['GPU Utilization (%)']:
        x = np.arange(len(names))
        width = 0.8 / max(1, max([len(gpus) for gpus in comparison_data['GPU Utilization (%)']]))
        
        for i, gpu_values in enumerate(comparison_data['GPU Utilization (%)']):
            for j, (gpu, value) in enumerate(gpu_values):
                ax1.bar(x[i] + j*width - 0.4 + width/2, value, width, 
                       label=f'GPU {gpu}' if i == 0 else "", 
                       color=COLORS[j % len(COLORS)])
        
        ax1.set_ylabel('사용률 (%)')
        ax1.set_title('GPU 사용률 비교')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        if any(gpu_values for gpu_values in comparison_data['GPU Utilization (%)']):
            ax1.legend()
    
    # 2. GPU 메모리 사용량 비교 차트 (오른쪽 상단)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if comparison_data['GPU Memory Used (MB)']:
        x = np.arange(len(names))
        width = 0.8 / max(1, max([len(gpus) for gpus in comparison_data['GPU Memory Used (MB)']]))
        
        for i, gpu_values in enumerate(comparison_data['GPU Memory Used (MB)']):
            for j, (gpu, value) in enumerate(gpu_values):
                ax2.bar(x[i] + j*width - 0.4 + width/2, value, width, 
                       label=f'GPU {gpu}' if i == 0 else "", 
                       color=COLORS[j % len(COLORS)])
        
        ax2.set_ylabel('메모리 (MB)')
        ax2.set_title('GPU 메모리 사용량 비교')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        if any(gpu_values for gpu_values in comparison_data['GPU Memory Used (MB)']):
            ax2.legend()
    
    # 3. GPU 전력 사용량 비교 차트 (왼쪽 하단)
    ax3 = fig.add_subplot(gs[1, 0])
    
    if comparison_data['GPU Power (W)']:
        x = np.arange(len(names))
        width = 0.8 / max(1, max([len(gpus) for gpus in comparison_data['GPU Power (W)']]))
        
        for i, gpu_values in enumerate(comparison_data['GPU Power (W)']):
            for j, (gpu, value) in enumerate(gpu_values):
                ax3.bar(x[i] + j*width - 0.4 + width/2, value, width, 
                       label=f'GPU {gpu}' if i == 0 else "", 
                       color=COLORS[j % len(COLORS)])
        
        ax3.set_ylabel('전력 (W)')
        ax3.set_title('GPU 전력 사용량 비교')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        
        if any(gpu_values for gpu_values in comparison_data['GPU Power (W)']):
            ax3.legend()
    
    # 4. GPU 효율성 비교 차트 (오른쪽 하단)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if (comparison_data['GPU Utilization (%)'] and 
        comparison_data['GPU Power (W)'] and 
        any(result_dfs[i]['Metric'].str.contains('Throughput').any() for i in range(len(result_dfs)) if 'Metric' in result_dfs[i].columns)):
        
        efficiency_data = []
        
        for i, df in enumerate(result_dfs):
            if 'Metric' not in df.columns or 'avg' not in df.columns:
                efficiency_data.append(0)
                continue
                
            throughput_row = df[df['Metric'] == 'Output Token Throughput (tokens/sec)']
            if throughput_row.empty:
                efficiency_data.append(0)
                continue
                
            throughput = clean_value(throughput_row.iloc[0]['avg'])
            
            # GPU 전력 사용량 평균 계산
            if comparison_data['GPU Power (W)'][i]:
                total_power = sum([value for _, value in comparison_data['GPU Power (W)'][i]])
                num_gpus = len(comparison_data['GPU Power (W)'][i])
                avg_power = total_power / num_gpus if num_gpus > 0 else 0
                
                # 효율성 = 처리량 / 전력
                efficiency = throughput / avg_power if avg_power > 0 else 0
                efficiency_data.append(efficiency)
            else:
                efficiency_data.append(0)
        
        if any(efficiency_data):
            ax4.bar(names, efficiency_data, color=COLORS[:len(names)])
            ax4.set_ylabel('토큰/초/와트')
            ax4.set_title('GPU 효율성 (처리량/전력)')
            ax4.set_xticklabels(names, rotation=45, ha='right')
            
            # 값 표시
            for i, v in enumerate(efficiency_data):
                ax4.text(i, v * 1.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return True
