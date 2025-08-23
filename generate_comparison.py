#!/usr/bin/env python3

import argparse
import os
import datetime
import time
from plot import compare_line_chart_from_csv
from plot import compare_plot_histogram
from evaluation import filter_and_export_csv
from evaluationC1 import filter_and_export_csv_C1

import pandas as pd

def add_final_percent_column_safe(csv_file, out_file):
    df = pd.read_csv(csv_file)
    # 重新计算，确保是数值
    ratio = (df.groupby(['num_tasks', 'per_jitter'])['final_e2e_max']
               .apply(lambda x: (x == 0).mean())
               .reset_index(name='finalpercent'))
    df = df.merge(ratio, on=['num_tasks', 'per_jitter'], how='left')

    # 强制转成 float，防止残留方法对象
    df['finalpercent'] = df['finalpercent'].astype(float)
    df.to_csv(out_file, index=False)
    return out_file


def generate_final_comparison(common_csv, common_csv_c1):

    if not os.path.exists(common_csv):
        print(f"error: can not found {common_csv}")
        return False
        
    if not os.path.exists(common_csv_c1):
        print(f"error: can not found {common_csv_c1}")
        return False
    

    output_dir = os.path.dirname(os.path.abspath(common_csv))


    data_output_dir = f"{output_dir}/data"

    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    common_csv = add_final_percent_column_safe(common_csv, common_csv)
    common_csv_c1 = add_final_percent_column_safe(common_csv_c1, common_csv_c1)
    
    csv_files = [common_csv, common_csv_c1]
    
    compare_percent_plot_name = os.path.join(output_dir, "final_compare_percent.png")
    compare_histogram_plot_name = os.path.join(output_dir, "final_compare_histogram.png")
    
    try:
        compare_line_chart_from_csv(csv_files, compare_percent_plot_name)
        compare_plot_histogram(csv_files, compare_histogram_plot_name)
        
        print(f"{compare_percent_plot_name} and {compare_histogram_plot_name} generated successfully")
        
        filter_and_export_csv(common_csv, [3, 5, 8, 10], data_output_dir)
        filter_and_export_csv_C1(common_csv_c1, [3, 5, 8, 10], data_output_dir)
        print(f"Filtered CSV files saved in {data_output_dir}")

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
    
    success = generate_final_comparison(args.common_csv, args.common_csv_c1)
    
    if success:
        print("Final comparison plot generated successfully!")
    else:
        print("ccurred while generating comparison plot!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())