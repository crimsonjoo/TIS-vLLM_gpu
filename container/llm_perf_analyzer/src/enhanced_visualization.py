#!/usr/bin/env python3
"""
Enhanced visualization module for LLM performance analysis

This module provides improved visualization functions for LLM performance test results,
focusing on delivering meaningful insights through better data representation.

Key features:
- Latency breakdown analysis
- Response time distribution analysis
- GPU efficiency visualization
- Performance summary chart
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to system path to access utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility modules
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
from utils.enhanced_charts import (
    create_latency_breakdown_chart,
    create_response_time_analysis,
    create_gpu_efficiency_chart,
    create_performance_summary_chart
)

def visualize_enhanced_results(result_dir, output_dir=None):
    """
    Visualize test results using enhanced visualization functions.
    
    Args:
        result_dir (str): Path to result directory
        output_dir (str, optional): Path to output directory. Defaults to None.
        
    Returns:
        bool: Success status
    """
    print(f"Generating enhanced visualizations from result directory '{result_dir}'...")
    
    # Find CSV file
    csv_file = os.path.join(result_dir, "raw_data", "profile_export_genai_perf.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return False
    
    # Parse CSV file
    try:
        print(f"Parsing CSV file '{csv_file}'...")
        
        # First section (latency metrics)
        latency_metrics = pd.read_csv(csv_file, nrows=6)
        
        # Second section (throughput metrics)
        throughput_metrics = pd.read_csv(csv_file, skiprows=8, nrows=3)
        
        # Third section (GPU metrics)
        gpu_metrics = pd.read_csv(csv_file, skiprows=13)
        
        # Combine all data into one DataFrame
        result_data = pd.concat([latency_metrics, throughput_metrics, gpu_metrics])
        print("CSV file parsing completed.")
    except Exception as e:
        print(f"Error: Failed to parse CSV file '{csv_file}': {e}")
        return False
    
    # Extract model name
    model_name = os.path.basename(result_dir).split('_')[1]  # perf_dna-r1-14B_20250407_050403 -> dna-r1-14B
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(result_dir, "charts")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set chart title
    title = f"{model_name} Performance Analysis"
    
    try:
        # 1. Create latency breakdown chart
        print("Creating latency breakdown analysis chart...")
        latency_breakdown_file = os.path.join(output_dir, f"{model_name}_latency_breakdown.png")
        create_latency_breakdown_chart(result_data, title, latency_breakdown_file)
        print(f"Latency breakdown analysis chart saved to '{latency_breakdown_file}'.")
        
        # 2. Create response time analysis chart
        print("Creating response time analysis chart...")
        response_time_file = os.path.join(output_dir, f"{model_name}_response_time.png")
        create_response_time_analysis(result_data, title, response_time_file)
        print(f"Response time analysis chart saved to '{response_time_file}'.")
        
        # 3. Create GPU efficiency chart
        print("Creating GPU efficiency analysis chart...")
        gpu_efficiency_file = os.path.join(output_dir, f"{model_name}_gpu_efficiency.png")
        create_gpu_efficiency_chart(result_data, title, gpu_efficiency_file)
        print(f"GPU efficiency analysis chart saved to '{gpu_efficiency_file}'.")
        
        # 4. Create performance summary chart
        print("Creating performance summary chart...")
        performance_summary_file = os.path.join(output_dir, f"{model_name}_performance_summary.png")
        create_performance_summary_chart(result_data, title, performance_summary_file)
        print(f"Performance summary chart saved to '{performance_summary_file}'.")
        
        print(f"Enhanced visualizations generated in '{output_dir}' directory.")
        return True
    except Exception as e:
        print(f"Error: Exception occurred during chart generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Enhanced visualization tool for LLM performance test results")
    parser.add_argument("result_dir", help="Path to result directory")
    parser.add_argument("--output-dir", help="Path to output directory")
    
    args = parser.parse_args()
    
    success = visualize_enhanced_results(args.result_dir, args.output_dir)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
