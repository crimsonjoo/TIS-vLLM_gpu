"""
# GenAI 성능 시각화를 위한 향상된 차트 생성 모듈

이 모듈은 LLM 성능 테스트 결과를 시각적으로 표현하기 위한 향상된 차트 생성 기능을 제공합니다.
다양한 CSV 구조와 누락된 데이터에 대해 견고하게 동작하도록 설계되어 있어, 다양한 테스트 환경에서
일관된 시각화 결과를 제공합니다.

## 주요 기능

1. 지연 시간 분석 차트 (Latency Breakdown Analysis)
   - 전체 지연 시간을 첫 번째 토큰 생성 시간, 토큰 생성 시간, 처리 오버헤드로 분해
   - 필요한 지표가 없는 경우 자동으로 추정값을 계산하여 차트 생성

2. 응답 시간 분석 차트 (Response Time Analysis)
   - 다양한 지연 시간 메트릭의 분포와 관계를 보여주는 종합적인 차트
   - 다양한 CSV 구조에 대응하여 누락된 지표가 있어도 적절한 대체값 사용

3. GPU 효율성 분석 차트 (GPU Efficiency Analysis)
   - GPU 사용률, 메모리 사용량, 전력 사용량 간의 관계 시각화
   - GPU 지표가 없는 경우에도 적절한 경고 메시지와 함께 차트 생성
   - NaN 값을 안전하게 처리하여 오류 없이 차트 생성

4. 성능 요약 차트 (Performance Summary)
   - 핵심 성능 메트릭과 인사이트를 한눈에 제공하는 종합적인 차트
   - 'Time To First Token'과 같은 중요 지표가 없는 경우 총 지연 시간의 일정 비율로 추정

## 사용 방법

```python
from utils.enhanced_charts import (
    create_latency_breakdown_chart,
    create_response_time_chart,
    create_gpu_efficiency_chart,
    create_performance_summary_chart
)

# CSV 파일에서 데이터 로드
df = pd.read_csv('path/to/csv')

# 차트 생성
create_latency_breakdown_chart(df, "모델명", "output_file.png")
```
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set visualization style
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
    Convert string values to numbers.
    
    Args:
        value: Value to convert
        
    Returns:
        float: Converted numeric value
    """
    if isinstance(value, str):
        if ',' in value:
            return float(value.replace(',', ''))
        try:
            return float(value)
        except ValueError:
            return 0.0
    return float(value) if value is not None else 0.0

def create_latency_breakdown_chart(df, title, output_file):
    """
    지연 시간 분석 차트를 생성합니다.
    
    이 차트는 전체 지연 시간을 첫 번째 토큰 생성 시간, 토큰 생성 시간, 처리 오버헤드로 분해하여 보여줍니다.
    각 구성 요소가 전체 지연 시간에 기여하는 비율을 백분율로 표시하며,
    필요한 지표가 없는 경우 자동으로 추정값을 계산하여 차트를 생성합니다.
    
    Args:
        df (DataFrame): 성능 지표가 포함된 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if df is None or df.empty:
        return False
    
    # Extract relevant metrics
    latency_metrics = df[df['Metric'].str.contains('Latency|Token', regex=True, na=False)]
    
    # Check if we have the required metrics
    if latency_metrics.empty or 'Request Latency (ms)' not in latency_metrics['Metric'].values:
        print(f"Warning: Required latency metrics not found in data for {title}")
        return False
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get total latency
    total_latency = clean_value(latency_metrics[latency_metrics['Metric'] == 'Request Latency (ms)']['avg'].values[0])
    
    # Get output sequence length for calculation
    output_length = clean_value(df[df['Metric'] == 'Output Sequence Length (tokens)']['avg'].values[0])
    
    # Since we don't have detailed breakdown, we'll estimate components
    # based on available metrics and industry standards
    
    # Estimate time to first token (typically 10-15% of total latency)
    ttft_estimate = total_latency * 0.15
    
    # Estimate token generation time (typically 70-80% of total latency)
    token_gen_estimate = total_latency * 0.75
    
    # Calculate remaining processing time
    remaining_time = total_latency - ttft_estimate - token_gen_estimate
    
    # Create stacked bar chart for latency breakdown
    components = ['Estimated Time to First Token', 'Estimated Token Generation Time', 'Processing Overhead']
    values = [ttft_estimate, token_gen_estimate, remaining_time]
    
    # Calculate percentages
    percentages = [v/total_latency*100 for v in values]
    
    # Plot
    ax = plt.subplot(111)
    bars = ax.bar(range(len(components)), values, color=COLORS[:3])
    
    # Add percentage and value labels
    for i, (bar, percentage, value) in enumerate(zip(bars, percentages, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{percentage:.1f}%\n({value:.1f} ms)',
                ha='center', va='bottom', fontsize=10)
    
    # Add annotations
    plt.title(f"Latency Breakdown Analysis - {title}")
    plt.ylabel('Time (ms)')
    plt.xticks(range(len(components)), components, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate estimated time per token
    estimated_time_per_token = token_gen_estimate / max(1, output_length - 1)  # avoid division by zero
    
    # Add insights text box
    textstr = '\n'.join((
        'INSIGHTS:',
        f'• Estimated first token response time: {ttft_estimate:.2f} ms ({percentages[0]:.1f}% of total)',
        f'• Estimated token generation time: {token_gen_estimate:.2f} ms ({percentages[1]:.1f}% of total)',
        f'• Processing overhead: {remaining_time:.2f} ms ({percentages[2]:.1f}% of total)',
        f'• Estimated time per token: {estimated_time_per_token:.2f} ms',
        f'• Total response time: {total_latency:.2f} ms for {output_length:.0f} tokens',
        f'• Output token throughput: {output_length/total_latency*1000:.2f} tokens/sec'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    plt.figtext(0.15, 0.02, textstr, fontsize=10, bbox=props)
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_response_time_chart(df, title, output_file):
    """
    응답 시간 분석 차트를 생성합니다.
    
    이 차트는 다양한 지연 시간 메트릭의 분포와 관계를 보여주는 종합적인 차트입니다.
    백분위수별 지연 시간 분포, 메트릭 간 비교, 처리량 분석, 시퀀스 길이와 지연 시간의 관계를 시각화합니다.
    다양한 CSV 구조에 대응하여 누락된 지표가 있어도 적절한 대체값을 사용합니다.
    
    Args:
        df (DataFrame): 성능 지표가 포함된 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if df is None or df.empty:
        return False
    
    # Extract relevant metrics
    latency_metrics = df[df['Metric'].str.contains('Latency|Token', regex=True, na=False)]
    
    # Check if we have the required metrics
    if latency_metrics.empty or 'Request Latency (ms)' not in latency_metrics['Metric'].values:
        print(f"Warning: Required latency metrics not found in data for {title}")
        return False
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Request Latency Percentile Distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    percentile_cols = ['min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']
    
    # Plot request latency percentiles
    request_latency = latency_metrics[latency_metrics['Metric'] == 'Request Latency (ms)']
    if not request_latency.empty:
        values = [clean_value(request_latency[col].values[0]) for col in percentile_cols if col in request_latency]
        if len(values) == len(percentile_cols):
            ax1.plot(percentile_cols, values, marker='o', label='Request Latency', color=COLORS[0], linewidth=2)
    
    ax1.set_title('Request Latency Percentile Distribution')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xlabel('Percentile')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Token Throughput (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Find throughput metrics in a different way
    # First, check if there's a separate section with 'Value' column
    throughput_section = None
    for i, row in df.iterrows():
        if row.get('Metric') == 'Output Token Throughput (per sec)':
            if 'Value' in df.columns and not pd.isna(row.get('Value')):
                throughput = clean_value(row['Value'])
                throughput_section = True
                break
            # If 'Value' column doesn't exist, check if the value is in 'avg' column
            elif 'avg' in df.columns and not pd.isna(row.get('avg')):
                throughput = clean_value(row['avg'])
                throughput_section = True
                break
    
    # If not found in either way, estimate from other metrics
    if not throughput_section:
        # Estimate throughput from request latency and output sequence length
        request_latency_val = clean_value(latency_metrics[latency_metrics['Metric'] == 'Request Latency (ms)']['avg'].values[0])
        output_length = clean_value(df[df['Metric'] == 'Output Sequence Length (tokens)']['avg'].values[0])
        throughput = output_length / (request_latency_val / 1000)  # convert ms to seconds
    
    ax2.bar(['Output Token Throughput'], [throughput], color=COLORS[1])
    ax2.text(0, throughput + 0.5, f'{throughput:.2f} tokens/sec', ha='center', va='bottom')
    
    ax2.set_title('Token Throughput')
    ax2.set_ylabel('Tokens per Second')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Sequence Length Distribution (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    sequence_metrics = ['Input Sequence Length (tokens)', 'Output Sequence Length (tokens)']
    seq_values = []
    seq_labels = []
    
    for metric in sequence_metrics:
        metric_row = df[df['Metric'] == metric]
        if not metric_row.empty:
            value = clean_value(metric_row['avg'].values[0])
            seq_values.append(value)
            seq_labels.append(metric.split(' (')[0])
    
    if seq_values:
        bars = ax3.bar(range(len(seq_labels)), seq_values, color=COLORS[2:2+len(seq_labels)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f} tokens',
                    ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Sequence Length Distribution')
    ax3.set_ylabel('Number of Tokens')
    ax3.set_xticks(range(len(seq_labels)))
    ax3.set_xticklabels(seq_labels)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Request Throughput (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Find request throughput in a similar way as token throughput
    req_throughput = 0
    for i, row in df.iterrows():
        if row.get('Metric') == 'Request Throughput (per sec)':
            if 'Value' in df.columns and not pd.isna(row.get('Value')):
                req_throughput = clean_value(row['Value'])
                break
            elif 'avg' in df.columns and not pd.isna(row.get('avg')):
                req_throughput = clean_value(row['avg'])
                break
    
    # If not found, estimate from request latency
    if req_throughput == 0:
        request_latency_val = clean_value(latency_metrics[latency_metrics['Metric'] == 'Request Latency (ms)']['avg'].values[0])
        req_throughput = 1000 / request_latency_val  # requests per second
    
    ax4.bar(['Request Throughput'], [req_throughput], color=COLORS[3])
    ax4.text(0, req_throughput + 0.02, f'{req_throughput:.2f} req/sec', ha='center', va='bottom')
    
    ax4.set_title('Request Throughput')
    ax4.set_ylabel('Requests per Second')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add insights text box
    request_latency_val = clean_value(latency_metrics[latency_metrics['Metric'] == 'Request Latency (ms)']['avg'].values[0])
    output_length = clean_value(df[df['Metric'] == 'Output Sequence Length (tokens)']['avg'].values[0])
    
    textstr = '\n'.join((
        'RESPONSE TIME INSIGHTS:',
        f'• Average request latency: {request_latency_val:.2f} ms',
        f'• Average output sequence length: {output_length:.1f} tokens',
        f'• Token generation rate: {output_length/request_latency_val*1000:.2f} tokens/sec',
        f'• Estimated time per token: {request_latency_val/output_length:.2f} ms/token'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    plt.figtext(0.5, 0.02, textstr, fontsize=10, bbox=props, ha='center')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.suptitle(f"Response Time Analysis - {title}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file)
    plt.close()
    
    return True

def create_gpu_efficiency_chart(df, title, output_file):
    """
    GPU 효율성 분석 차트를 생성합니다.
    
    이 차트는 GPU 사용률, 메모리 사용량, 전력 사용량 간의 관계를 시각화합니다.
    GPU 리소스 활용도와 성능 메트릭 간의 상관관계를 보여주며,
    GPU 지표가 없는 경우에도 적절한 경고 메시지와 함께 차트를 생성합니다.
    NaN 값을 안전하게 처리하여 오류 없이 차트를 생성합니다.
    
    Args:
        df (DataFrame): 성능 지표가 포함된 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        if df is None or df.empty:
            print(f"Warning: Empty dataframe provided for {title}")
            return create_empty_gpu_efficiency_chart(title, output_file)
        
        # 기본 데이터 확인
        has_gpu_metrics = 'GPU' in df.columns
        gpu_metrics = pd.DataFrame()
        
        if has_gpu_metrics:
            # Extract GPU metrics
            gpu_metrics = df[df['Metric'].str.contains('GPU', na=False)]
            
        if not has_gpu_metrics or gpu_metrics.empty:
            print(f"Warning: GPU metrics not found in data for {title}")
            return create_empty_gpu_efficiency_chart(title, output_file)
        
        # Create figure with subplots - 해상도 높이고 여백 조정
        plt.figure(figsize=(14, 10), dpi=150)
        gs = GridSpec(3, 2, height_ratios=[0.2, 1, 1], width_ratios=[1, 1], 
                    hspace=0.3, wspace=0.25)
        
        # GPU 번호별 색상 매핑 생성
        # 모든 GPU 지표에서 고유한 GPU ID 추출
        all_gpu_data = gpu_metrics[~gpu_metrics['GPU'].isna()]
        all_gpu_ids = sorted(all_gpu_data['GPU'].unique())
        
        # GPU ID별 색상 매핑 생성 (각 GPU가 모든 차트에서 동일한 색상을 가지도록)
        gpu_colors = {}
        for i, gpu_id in enumerate(all_gpu_ids):
            gpu_colors[gpu_id] = COLORS[i % len(COLORS)]
        
        # 인사이트 텍스트 박스 (상단에 배치)
        ax_insights = plt.subplot(gs[0, :])
        ax_insights.axis('off')  # 축 숨기기
        
        # 1. GPU Utilization by GPU (중간 왼쪽)
        ax1 = plt.subplot(gs[1, 0])
        
        utilization_data = gpu_metrics[gpu_metrics['Metric'] == 'GPU Utilization (%)']
        if not utilization_data.empty:
            gpu_ids = utilization_data['GPU'].unique()
            
            # NaN 값 안전하게 처리
            util_values = []
            for gpu in gpu_ids:
                try:
                    value = utilization_data[utilization_data['GPU'] == gpu]['avg'].values[0]
                    util_values.append(clean_value(value))
                except (IndexError, ValueError):
                    util_values.append(0.0)
            
            # GPU ID별로 지정된 색상 사용
            colors = [gpu_colors.get(gpu, COLORS[0]) for gpu in gpu_ids]
            bars = ax1.bar(gpu_ids, util_values, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
            
            ax1.set_title('GPU Utilization')
            ax1.set_ylabel('Utilization (%)')
            ax1.set_xlabel('GPU ID')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.set_ylim(0, max(util_values) * 1.15 if util_values else 105)  # 동적 y축 범위 설정
        else:
            ax1.text(0.5, 0.5, 'GPU Utilization Data Not Available', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax1.transAxes, style='italic', color='gray')
            ax1.set_title('GPU Utilization')
            ax1.set_ylabel('Utilization (%)')
            ax1.set_xlabel('GPU ID')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. GPU Power Usage (중간 오른쪽)
        ax2 = plt.subplot(gs[1, 1])
        
        power_data = gpu_metrics[gpu_metrics['Metric'] == 'GPU Power Usage (W)']
        if not power_data.empty:
            gpu_ids = power_data['GPU'].unique()
            
            # NaN 값 안전하게 처리
            power_values = []
            for gpu in gpu_ids:
                try:
                    value = power_data[power_data['GPU'] == gpu]['avg'].values[0]
                    power_values.append(clean_value(value))
                except (IndexError, ValueError):
                    power_values.append(0.0)
            
            # GPU ID별로 지정된 색상 사용
            colors = [gpu_colors.get(gpu, COLORS[0]) for gpu in gpu_ids]
            bars = ax2.bar(gpu_ids, power_values, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f} W',
                        ha='center', va='bottom', fontsize=9)
            
            ax2.set_title('GPU Power Usage')
            ax2.set_ylabel('Power (W)')
            ax2.set_xlabel('GPU ID')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.set_ylim(0, max(power_values) * 1.15 if power_values else 100)  # 동적 y축 범위 설정
        else:
            ax2.text(0.5, 0.5, 'GPU Power Usage Data Not Available', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax2.transAxes, style='italic', color='gray')
            ax2.set_title('GPU Power Usage')
            ax2.set_ylabel('Power (W)')
            ax2.set_xlabel('GPU ID')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. GPU Memory Usage (하단 왼쪽)
        ax3 = plt.subplot(gs[2, 0])
        
        memory_data = gpu_metrics[gpu_metrics['Metric'] == 'GPU Memory Used (GB)']
        if not memory_data.empty:
            gpu_ids = memory_data['GPU'].unique()
            
            # NaN 값 안전하게 처리
            memory_values = []
            for gpu in gpu_ids:
                try:
                    value = memory_data[memory_data['GPU'] == gpu]['avg'].values[0]
                    memory_values.append(clean_value(value))
                except (IndexError, ValueError):
                    memory_values.append(0.0)
            
            # GPU ID별로 지정된 색상 사용
            colors = [gpu_colors.get(gpu, COLORS[0]) for gpu in gpu_ids]
            bars = ax3.bar(gpu_ids, memory_values, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f} GB',
                        ha='center', va='bottom', fontsize=9)
            
            ax3.set_title('GPU Memory Usage')
            ax3.set_ylabel('Memory (GB)')
            ax3.set_xlabel('GPU ID')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
            ax3.set_ylim(0, max(memory_values) * 1.15 if memory_values else 100)  # 동적 y축 범위 설정
        else:
            ax3.text(0.5, 0.5, 'GPU Memory Usage Data Not Available', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax3.transAxes, style='italic', color='gray')
            ax3.set_title('GPU Memory Usage')
            ax3.set_ylabel('Memory (GB)')
            ax3.set_xlabel('GPU ID')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Energy Efficiency Analysis (하단 오른쪽)
        ax4 = plt.subplot(gs[2, 1])
        
        # Extract throughput metrics for efficiency calculation
        token_throughput = 0
        try:
            # 먼저 'Value' 열에서 찾기 시도
            if 'Value' in df.columns and not df[df['Metric'] == 'Output Token Throughput (per sec)']['Value'].empty:
                token_throughput = clean_value(df[df['Metric'] == 'Output Token Throughput (per sec)']['Value'].values[0])
            # 'Value' 열에 없으면 'avg' 열에서 찾기 시도
            elif 'avg' in df.columns and not df[df['Metric'] == 'Output Token Throughput (per sec)']['avg'].empty:
                token_throughput = clean_value(df[df['Metric'] == 'Output Token Throughput (per sec)']['avg'].values[0])
            # 그래도 없으면 다른 메트릭에서 계산 시도
            else:
                # 요청 지연 시간과 출력 시퀀스 길이를 사용하여 추정
                if 'Request Latency (ms)' in df['Metric'].values and 'Output Sequence Length (tokens)' in df['Metric'].values:
                    request_latency = clean_value(df[df['Metric'] == 'Request Latency (ms)']['avg'].values[0])
                    output_length = clean_value(df[df['Metric'] == 'Output Sequence Length (tokens)']['avg'].values[0])
                    if request_latency > 0:
                        token_throughput = output_length / (request_latency / 1000)  # 초당 토큰 수
        except (IndexError, KeyError, ValueError) as e:
            print(f"Warning: Could not extract token throughput for {title}: {e}")
            token_throughput = 0
        
        if not power_data.empty and token_throughput > 0:
            gpu_ids = power_data['GPU'].unique()
            
            # NaN 값 안전하게 처리
            power_values = []
            for gpu in gpu_ids:
                try:
                    value = power_data[power_data['GPU'] == gpu]['avg'].values[0]
                    power_values.append(clean_value(value))
                except (IndexError, ValueError):
                    power_values.append(0.1)  # 0으로 나누기 방지를 위해 0.1 사용
            
            # Calculate tokens per watt - 0으로 나누기 방지
            tokens_per_watt = [token_throughput / max(0.1, power) for power in power_values]
            
            # GPU ID별로 지정된 색상 사용
            colors = [gpu_colors.get(gpu, COLORS[0]) for gpu in gpu_ids]
            bars = ax4.bar(gpu_ids, tokens_per_watt, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height * 0.05,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
            
            ax4.set_title('Energy Efficiency (Tokens per Watt)')
            ax4.set_ylabel('Tokens/Watt')
            ax4.set_xlabel('GPU ID')
            ax4.grid(axis='y', linestyle='--', alpha=0.7)
            ax4.set_ylim(0, max(tokens_per_watt) * 1.15 if tokens_per_watt else 10)  # 동적 y축 범위 설정
        else:
            # 데이터가 없는 경우 메시지 표시
            ax4.text(0.5, 0.5, 'Energy Efficiency Data Not Available', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax4.transAxes, style='italic', color='gray')
            ax4.set_title('Energy Efficiency (Tokens per Watt)')
            ax4.set_ylabel('Tokens/Watt')
            ax4.set_xlabel('GPU ID')
            ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add insights text box at the top with larger font
        insights_text = 'GPU EFFICIENCY INSIGHTS'
        
        if not utilization_data.empty or not power_data.empty or not memory_data.empty:
            try:
                # 사용 가능한 데이터만 활용하여 인사이트 생성
                insights = []
                
                if not utilization_data.empty:
                    gpu_ids = utilization_data['GPU'].unique()
                    util_values = []
                    for gpu in gpu_ids:
                        try:
                            if gpu in utilization_data['GPU'].values:
                                value = utilization_data[utilization_data['GPU'] == gpu]['avg'].values[0]
                                util_values.append(clean_value(value))
                        except (IndexError, ValueError):
                            pass
                    
                    if util_values:
                        avg_util = np.mean(util_values)
                        insights.append(f'• 평균 GPU 사용률: {avg_util:.1f}%')
                        
                        # 가장 높은/낮은 사용률 GPU 찾기
                        util_dict = {}
                        for i, gpu in enumerate(gpu_ids):
                            if i < len(util_values):
                                util_dict[gpu] = util_values[i]
                        
                        if util_dict:
                            max_util_gpu = max(util_dict, key=util_dict.get)
                            min_util_gpu = min(util_dict, key=util_dict.get)
                            insights.append(f'• 최대 사용률 GPU: {max_util_gpu} ({util_dict[max_util_gpu]:.1f}%)')
                            insights.append(f'• 최소 사용률 GPU: {min_util_gpu} ({util_dict[min_util_gpu]:.1f}%)')
                
                if not power_data.empty:
                    gpu_ids = power_data['GPU'].unique()
                    power_values = []
                    for gpu in gpu_ids:
                        try:
                            if gpu in power_data['GPU'].values:
                                value = power_data[power_data['GPU'] == gpu]['avg'].values[0]
                                power_values.append(clean_value(value))
                        except (IndexError, ValueError):
                            pass
                    
                    if power_values:
                        avg_power = np.mean(power_values)
                        insights.append(f'• 평균 전력 사용량: {avg_power:.1f} W')
                
                if not memory_data.empty:
                    gpu_ids = memory_data['GPU'].unique()
                    memory_values = []
                    for gpu in gpu_ids:
                        try:
                            if gpu in memory_data['GPU'].values:
                                value = memory_data[memory_data['GPU'] == gpu]['avg'].values[0]
                                memory_values.append(clean_value(value))
                        except (IndexError, ValueError):
                            pass
                    
                    if memory_values:
                        avg_memory = np.mean(memory_values)
                        insights.append(f'• 평균 메모리 사용량: {avg_memory:.1f} GB')
                
                if token_throughput > 0:
                    insights.append(f'• 토큰 처리량: {token_throughput:.2f} tokens/sec')
                    
                    if not power_data.empty and power_values:
                        avg_power = np.mean(power_values)
                        if avg_power > 0:
                            efficiency = token_throughput / avg_power
                            insights.append(f'• 에너지 효율성: {efficiency:.2f} tokens/watt')
                
                # 인사이트 텍스트 생성
                if insights:
                    insights_text = 'GPU EFFICIENCY INSIGHTS\n' + '\n'.join(insights)
                
            except Exception as e:
                print(f"Warning: Error generating GPU insights for {title}: {e}")
                insights_text = 'GPU EFFICIENCY INSIGHTS\n• 데이터 부족으로 상세 인사이트를 생성할 수 없습니다.'
        else:
            insights_text = 'GPU EFFICIENCY INSIGHTS\n• 데이터 부족으로 상세 인사이트를 생성할 수 없습니다.'
        
        # 인사이트 텍스트 박스 표시 (왼쪽 정렬)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax_insights.text(0.01, 0.5, insights_text, fontsize=12, fontweight='bold', 
                        ha='left', va='center', transform=ax_insights.transAxes,
                        bbox=props)
        
        plt.suptitle(f"GPU Efficiency Analysis - {title}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 이미지 저장 - 해상도 높이고 투명도 제거
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error in create_gpu_efficiency_chart for {title}: {e}")
        # 오류 발생 시에도 빈 차트라도 생성
        return create_empty_gpu_efficiency_chart(title, output_file)

def create_empty_gpu_efficiency_chart(title, output_file):
    """
    GPU 메트릭 데이터가 없을 때 기본 GPU 효율성 차트를 생성합니다.
    
    Args:
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 기본 차트 생성 - 해상도 향상
        plt.figure(figsize=(14, 10), dpi=150)
        gs = GridSpec(3, 2, height_ratios=[0.2, 1, 1], width_ratios=[1, 1], 
                    hspace=0.3, wspace=0.25)
        
        # 인사이트 텍스트 박스 (상단에 배치)
        ax_insights = plt.subplot(gs[0, :])
        ax_insights.axis('off')  # 축 숨기기
        
        # 1. GPU Utilization by GPU (중간 왼쪽)
        ax1 = plt.subplot(gs[1, 0])
        ax1.text(0.5, 0.5, 'GPU Utilization Data Not Available', 
                ha='center', va='center', fontsize=12, 
                transform=ax1.transAxes, style='italic', color='gray')
        ax1.set_title('GPU Utilization')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_xlabel('GPU ID')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. GPU Power Usage (중간 오른쪽)
        ax2 = plt.subplot(gs[1, 1])
        ax2.text(0.5, 0.5, 'GPU Power Usage Data Not Available', 
                ha='center', va='center', fontsize=12, 
                transform=ax2.transAxes, style='italic', color='gray')
        ax2.set_title('GPU Power Usage')
        ax2.set_ylabel('Power (W)')
        ax2.set_xlabel('GPU ID')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. GPU Memory Usage (하단 왼쪽)
        ax3 = plt.subplot(gs[2, 0])
        ax3.text(0.5, 0.5, 'GPU Memory Usage Data Not Available', 
                ha='center', va='center', fontsize=12, 
                transform=ax3.transAxes, style='italic', color='gray')
        ax3.set_title('GPU Memory Usage')
        ax3.set_ylabel('Memory (GB)')
        ax3.set_xlabel('GPU ID')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Energy Efficiency Analysis (하단 오른쪽)
        ax4 = plt.subplot(gs[2, 1])
        ax4.text(0.5, 0.5, 'Energy Efficiency Data Not Available', 
                ha='center', va='center', fontsize=12, 
                transform=ax4.transAxes, style='italic', color='gray')
        ax4.set_title('Energy Efficiency (Tokens per Watt)')
        ax4.set_ylabel('Tokens/Watt')
        ax4.set_xlabel('GPU ID')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 인사이트 텍스트 박스 표시 (왼쪽 정렬)
        insights_text = 'GPU EFFICIENCY INSIGHTS\n• GPU 메트릭 데이터가 없어 인사이트를 생성할 수 없습니다.'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax_insights.text(0.01, 0.5, insights_text, fontsize=12, fontweight='bold', 
                        ha='left', va='center', transform=ax_insights.transAxes,
                        bbox=props)
        
        plt.suptitle(f"GPU Efficiency Analysis - {title}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 이미지 저장 - 해상도 높이고 투명도 제거
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error in create_empty_gpu_efficiency_chart for {title}: {e}")
        # 최소한의 빈 이미지라도 생성
        plt.figure(figsize=(14, 10))
        plt.text(0.5, 0.5, f"GPU Efficiency Analysis - {title}\n\nError generating chart: {str(e)}", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_file, dpi=150)
        plt.close()
        return False

def create_performance_summary_chart(df, title, output_file):
    """
    성능 요약 차트를 생성합니다.
    
    이 차트는 핵심 성능 메트릭과 인사이트를 한눈에 제공하는 종합적인 차트입니다.
    주요 지연 시간 메트릭, 처리량, 효율성 지표를 종합적으로 보여주며,
    'Time To First Token'과 같은 중요 지표가 없는 경우 총 지연 시간의 일정 비율(15%)로 추정하여 표시합니다.
    추정된 값은 차트에 명확하게 표시되어 사용자가 인지할 수 있습니다.
    
    Args:
        df (DataFrame): 성능 지표가 포함된 데이터프레임
        title (str): 차트 제목
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if df is None or df.empty:
        return False
    
    # Check if we have the required metrics
    if 'Request Latency (ms)' not in df['Metric'].values:
        print(f"Warning: Required latency metrics not found in data for {title}")
        return False
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Extract key metrics
    # Use available metrics or estimate if not available
    request_latency = clean_value(df[df['Metric'] == 'Request Latency (ms)']['avg'].values[0])
    output_length = clean_value(df[df['Metric'] == 'Output Sequence Length (tokens)']['avg'].values[0])
    
    # Estimate TTFT if not available (typically 10-15% of total latency)
    ttft_row = df[df['Metric'] == 'Time To First Token (ms)']
    if ttft_row.empty:
        ttft = request_latency * 0.15  # Estimate as 15% of total latency
        ttft_is_estimate = True
    else:
        ttft = clean_value(ttft_row['avg'].values[0])
        ttft_is_estimate = False
    
    # Find or estimate token throughput
    token_throughput = None
    for i, row in df.iterrows():
        if row.get('Metric') == 'Output Token Throughput (per sec)':
            if 'Value' in df.columns and not pd.isna(row.get('Value')):
                token_throughput = clean_value(row['Value'])
                break
            elif 'avg' in df.columns and not pd.isna(row.get('avg')):
                token_throughput = clean_value(row['avg'])
                break
    
    # If not found, estimate from request latency and output sequence length
    if token_throughput is None:
        token_throughput = output_length / (request_latency / 1000)  # convert ms to seconds
    
    # 1. Key Performance Indicators (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create KPI metrics
    kpi_metrics = [
        'Response Time',
        'First Token Time' if not ttft_is_estimate else 'Est. First Token Time',
        'Token Throughput'
    ]
    
    kpi_values = [
        request_latency,
        ttft,
        token_throughput
    ]
    
    kpi_units = [
        'ms',
        'ms',
        'tokens/sec'
    ]
    
    # Create horizontal bar chart for KPIs
    y_pos = np.arange(len(kpi_metrics))
    ax1.barh(y_pos, kpi_values, color=COLORS[:len(kpi_metrics)])
    
    # Add value labels
    for i, (value, unit) in enumerate(zip(kpi_values, kpi_units)):
        ax1.text(value + (max(kpi_values) * 0.02), i, f'{value:.2f} {unit}', va='center')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(kpi_metrics)
    ax1.set_title('Key Performance Indicators')
    ax1.set_xlabel('Value')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Sequence Length vs Response Time (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create a simple visualization of sequence length vs response time
    ax2.bar(['Input Tokens', 'Output Tokens'], 
            [clean_value(df[df['Metric'] == 'Input Sequence Length (tokens)']['avg'].values[0]),
             output_length], 
            color=[COLORS[3], COLORS[4]])
    
    # Add a secondary y-axis for response time
    ax3 = ax2.twinx()
    ax3.plot(['Input Tokens', 'Output Tokens'], [0, request_latency], 'r-o', label='Response Time')
    ax3.set_ylabel('Response Time (ms)', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Sequence Length vs Response Time')
    ax2.set_ylabel('Number of Tokens')
    
    # 3. Efficiency Metrics (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate efficiency metrics
    tokens_per_ms = token_throughput / 1000  # tokens per millisecond
    ms_per_token = 1 / tokens_per_ms if tokens_per_ms > 0 else 0
    
    efficiency_metrics = [
        'Tokens per Second',
        'Milliseconds per Token',
        'Tokens per Request'
    ]
    
    efficiency_values = [
        token_throughput,
        ms_per_token,
        output_length
    ]
    
    efficiency_units = [
        'tokens/sec',
        'ms/token',
        'tokens'
    ]
    
    # Create horizontal bar chart for efficiency metrics
    y_pos = np.arange(len(efficiency_metrics))
    ax4.barh(y_pos, efficiency_values, color=COLORS[5:5+len(efficiency_metrics)])
    
    # Add value labels
    for i, (value, unit) in enumerate(zip(efficiency_values, efficiency_units)):
        ax4.text(value + (max(efficiency_values) * 0.02), i, f'{value:.2f} {unit}', va='center')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(efficiency_metrics)
    ax4.set_title('Efficiency Metrics')
    ax4.set_xlabel('Value')
    ax4.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 4. Latency Percentiles (bottom right)
    ax5 = fig.add_subplot(gs[1, 1])
    
    percentile_cols = ['min', 'p50', 'p90', 'p95', 'p99', 'max']
    request_latency_row = df[df['Metric'] == 'Request Latency (ms)']
    
    if not request_latency_row.empty:
        percentile_values = [clean_value(request_latency_row[col].values[0]) for col in percentile_cols if col in request_latency_row]
        if len(percentile_values) == len(percentile_cols):
            ax5.plot(percentile_cols, percentile_values, 'b-o', linewidth=2)
            
            # Add value labels
            for i, value in enumerate(percentile_values):
                ax5.text(i, value + (max(percentile_values) * 0.02), f'{value:.2f} ms', ha='center')
    
    ax5.set_title('Latency Percentiles')
    ax5.set_ylabel('Time (ms)')
    ax5.set_xlabel('Percentile')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # Add insights text box
    if ttft_is_estimate:
        ttft_text = f'• Estimated first token response time: {ttft:.2f} ms ({ttft/request_latency*100:.1f}% of total)'
    else:
        ttft_text = f'• First token response time: {ttft:.2f} ms ({ttft/request_latency*100:.1f}% of total)'
    
    textstr = '\n'.join((
        'PERFORMANCE SUMMARY:',
        f'• Average response time: {request_latency:.2f} ms for {output_length:.1f} tokens',
        ttft_text,
        f'• Token generation rate: {token_throughput:.2f} tokens/sec',
        f'• Average time per token: {ms_per_token:.2f} ms/token',
        f'• P99/P50 latency ratio: {clean_value(request_latency_row["p99"].values[0])/clean_value(request_latency_row["p50"].values[0]):.2f}x (lower is better)'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    plt.figtext(0.5, 0.01, textstr, fontsize=10, bbox=props, ha='center')
    
    plt.suptitle(f"Performance Summary - {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    
    return True
