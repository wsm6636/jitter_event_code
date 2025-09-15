#!/usr/bin/env python3

import argparse
import os
import datetime
import time
from plot import compare_line_chart_from_csv
from plot import compare_plot_histogram


def generate_final_comparison(common_csv, common_csv_c1, output_dir=None):

    if not os.path.exists(common_csv):
        print(f"error: can not found {common_csv}")
        return False
        
    if not os.path.exists(common_csv_c1):
        print(f"error: can not found {common_csv_c1}")
        return False
    
    if output_dir is None:
        timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")
        output_dir = f"final_comparison_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [common_csv, common_csv_c1]
    
    compare_percent_plot_name = os.path.join(output_dir, "final_compare_percent.png")
    compare_histogram_plot_name = os.path.join(output_dir, "final_compare_histogram.png")
    
    try:
        compare_line_chart_from_csv(csv_files, compare_percent_plot_name)
        compare_plot_histogram(csv_files, compare_histogram_plot_name)
        
        print(f"{compare_percent_plot_name} and {compare_histogram_plot_name} generated successfully")
        return True
        
    except Exception as e:
        print(f"error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='generate final comparison plots from CSV files')
    parser.add_argument('--common_csv', type=str, default='common_results.csv',
                        help='rtss result csv file (common_results.csv)')
    parser.add_argument('--common_csv_c1', type=str, default='common_results_c1.csv',
                        help='C1 result csv file (common_results_c1.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='output path')
    
    args = parser.parse_args()
    
    if os.path.exists(args.common_csv):
        with open(args.common_csv, 'r') as f:
            lines = len(f.readlines()) - 1  
        print(f"rtss result total rows: {lines}")
    else:
        print(f"can not find {args.common_csv}")
    
    if os.path.exists(args.common_csv_c1):
        with open(args.common_csv_c1, 'r') as f:
            lines = len(f.readlines()) - 1  
        print(f"C1 result total rows: {lines}")
    else:
        print(f"can not find {args.common_csv_c1}")
    
    success = generate_final_comparison(args.common_csv, args.common_csv_c1, args.output_dir)
    
    if success:
        print("Final comparison plot generated successfully!")
    else:
        print("ccurred while generating comparison plot!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())