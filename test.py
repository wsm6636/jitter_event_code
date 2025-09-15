# boxplot_numtasks_our_G.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读原始 CSV
df = pd.read_csv('compare/20250909_155538_IC/common_results_passive_IC_20250909_155538_processed.csv')          # 文件放同目录即可

# 2. 把 5 数概括整理成“长表”，方便 seaborn 直接画箱线图
records = []
for _, row in df.iterrows():
    nt = row['num_tasks']
    # our 组
    records.extend([
        {'num_tasks': nt, 'method': 'our', 'value': row['run_time_our_min']},
        {'num_tasks': nt, 'method': 'our', 'value': row['run_time_our_q1']},
        {'num_tasks': nt, 'method': 'our', 'value': row['run_time_our_median']},
        {'num_tasks': nt, 'method': 'our', 'value': row['run_time_our_q3']},
        {'num_tasks': nt, 'method': 'our', 'value': row['run_time_our_max']},
    ])
    # G 组
    records.extend([
        {'num_tasks': nt, 'method': 'G', 'value': row['run_time_G_min']},
        {'num_tasks': nt, 'method': 'G', 'value': row['run_time_G_q1']},
        {'num_tasks': nt, 'method': 'G', 'value': row['run_time_G_median']},
        {'num_tasks': nt, 'method': 'G', 'value': row['run_time_G_q3']},
        {'num_tasks': nt, 'method': 'G', 'value': row['run_time_G_max']},
    ])

long_df = pd.DataFrame(records)

# 3. 画图
plt.figure(figsize=(6,4))
sns.boxplot(
    data=long_df,
    x='num_tasks',
    y='value',
    hue='method',
    palette={'our': 'tomato', 'G': 'steelblue'},
    saturation=1,
    width=0.7,
    fliersize=0          # 去掉离群点符号，因为 5 数概括已含极值
)
plt.yscale('log')
plt.xlabel('num_tasks')
plt.ylabel('run_time (s)')
plt.title('Boxplot: our vs G by num_tasks')
plt.tight_layout()
plt.savefig('boxplot_numtasks_our_G.png', dpi=300)
plt.show()