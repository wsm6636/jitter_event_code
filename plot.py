import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse 
from scipy.stats import gaussian_kde

def plot_histogram_from_csv(csv_file,R_plot_name):
    jitter_to_r_values = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        # r_values_count = 0  
        for row in reader:
            per_jitter = float(row['per_jitter'])
            r_value = float(row['R']) if row['R'] else None
            if r_value is not None:
                if per_jitter not in jitter_to_r_values:
                    jitter_to_r_values[per_jitter] = []
                jitter_to_r_values[per_jitter].append(r_value)
                if r_value > 1:
                    print(f"Warning: R value {r_value} exceeds 1.0 for per_jitter={per_jitter}. This may indicate an error in the data.")
                # r_values_count += 1  

    plt.figure(figsize=(12, 8))
    num_bins = 50
    bin_range = (0, 1.05)  
    bin_width = (bin_range[1] - bin_range[0]) / num_bins

    jitters = sorted(jitter_to_r_values.keys())
    num_groups = len(jitters)
    group_width = bin_width / (num_groups + 1)  
    # print(f"Number of valid R values: {r_values_count}")  
    for idx, per_jitter in enumerate(jitters):
        r_values = jitter_to_r_values[per_jitter]
        
        # Grouped Histogram
        offset = (idx - num_groups / 2) * group_width
        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers + offset, counts, width=group_width, alpha=0.3, align='center')

        # trend line
        if len(r_values) > 1:  
            kde = gaussian_kde(r_values)
            x = np.linspace(bin_range[0], bin_range[1], 1000)
            y = kde(x)
            plt.plot(x, y * len(r_values) * bin_width, linestyle='-', label=f'per_jitter={per_jitter}',color=plt.cm.tab10(idx))
        else:
            print(f"Skipping KDE for per_jitter={per_jitter} due to insufficient data points.")
    

    plt.title("Distribution of R Values for Different Jitter Percentages")
    plt.xlabel("R = max_reaction_time / final_e2e_max")
    plt.ylabel("Frequency")
    plt.xlim(bin_range)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{R_plot_name}")
    # plt.show()

    


def plot_line_chart_from_csv(csv_file, percent_plot_name):
    jitter_to_false_percentage = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            false_percentage = float(row['false_percentage'])  
            num_tasks = int(row['num_tasks'])  

            if num_tasks not in jitter_to_false_percentage:
                jitter_to_false_percentage[num_tasks] = {}

            if per_jitter not in jitter_to_false_percentage[num_tasks]:
                jitter_to_false_percentage[num_tasks][per_jitter] = []

            jitter_to_false_percentage[num_tasks][per_jitter].append(false_percentage)

    plt.figure(figsize=(10, 6))
    for num_tasks, jitter_data in jitter_to_false_percentage.items():
        jitter_percent = [jitter * 100 for jitter in sorted(jitter_data.keys())]
        false_percentages = [np.mean(jitter_data[jitter]) for jitter in sorted(jitter_data.keys())]
        plt.plot(jitter_percent, false_percentages, label=f"num_tasks={num_tasks}", marker='o')

    plt.title("False Percentage vs. Jitter for Different Number of Tasks")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("False Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  
    plt.savefig(f"{percent_plot_name}")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("R_plot_name", type=str, help="Name of the output plot file for R values.")
    parser.add_argument("percent_plot_name", type=str, help="Name of the output plot file for false percentages.")
    args = parser.parse_args()

    plot_histogram_from_csv(args.csv_file, args.R_plot_name)
    plot_line_chart_from_csv(args.csv_file, args.percent_plot_name)
