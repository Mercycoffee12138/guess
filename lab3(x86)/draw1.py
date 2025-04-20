import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置更现代的绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# 自定义颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
sequential_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
cmap = LinearSegmentedColormap.from_list("custom_blue", sequential_colors)

# 读取三个CSV文件
df_o0 = pd.read_csv('md5_performance_O0.csv')
df_o1 = pd.read_csv('md5_performance_O1.csv')
df_o2 = pd.read_csv('md5_performance_O2.csv')

# 获取汇总数据(总计行)
summary_o0 = df_o0[df_o0['批次'] == '总计'].iloc[0]
summary_o1 = df_o1[df_o1['批次'] == '总计'].iloc[0]
summary_o2 = df_o2[df_o2['批次'] == '总计'].iloc[0]

# 获取每秒处理密码数行
speed_o0 = df_o0[df_o0['批次'] == '每秒处理密码数'].iloc[0]
speed_o1 = df_o1[df_o1['批次'] == '每秒处理密码数'].iloc[0]
speed_o2 = df_o2[df_o2['批次'] == '每秒处理密码数'].iloc[0]

# 只保留批次为数字的行（去掉汇总行）
df_o0 = df_o0[pd.to_numeric(df_o0['批次'], errors='coerce').notna()]
df_o1 = df_o1[pd.to_numeric(df_o1['批次'], errors='coerce').notna()]
df_o2 = df_o2[pd.to_numeric(df_o2['批次'], errors='coerce').notna()]

# 准备数据
methods = ['串行', '二路SIMD', '四路SIMD', '八路SIMD']
opt_levels = ['O0 (无优化)', 'O1 (一级优化)', 'O2 (二级优化)']
speed_o0_values = [float(speed_o0[2]), float(speed_o0[3]), float(speed_o0[4]), float(speed_o0[5])]
speed_o1_values = [float(speed_o1[2]), float(speed_o1[3]), float(speed_o1[4]), float(speed_o1[5])]
speed_o2_values = [float(speed_o2[2]), float(speed_o2[3]), float(speed_o2[4]), float(speed_o2[5])]

# 创建图1：使用更高级的折线图来展示不同优化级别和SIMD路数下的性能
plt.figure(figsize=(14, 8))
markers = ['o', 's', 'd', '^']
line_styles = ['-', '--', '-.', ':']

for i, method in enumerate(methods):
    plt.plot([0, 1, 2], 
             [speed_o0_values[i]/1e6, speed_o1_values[i]/1e6, speed_o2_values[i]/1e6],
             marker=markers[i], markersize=10, linewidth=2.5, 
             linestyle=line_styles[i], color=colors[i], label=method)

plt.xticks([0, 1, 2], opt_levels)
plt.xlabel('编译优化级别')
plt.ylabel('处理速度 (百万密码/秒)')
plt.title('不同编译优化级别和SIMD实现下的密码处理性能')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)

# 添加数据标签
for i, method in enumerate(methods):
    for j, opt in enumerate([0, 1, 2]):
        speeds = [speed_o0_values, speed_o1_values, speed_o2_values]
        plt.annotate(f'{speeds[j][i]/1e6:.2f}M', 
                    xy=(opt, speeds[j][i]/1e6),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('advanced_performance_comparison.png', dpi=300)
plt.close()

# 创建图2：使用改进的柱状图和折线图组合展示加速比
plt.figure(figsize=(14, 8))
speedup_types = ['二路加速比', '四路加速比', '八路加速比']

data = {
    'O0': [float(summary_o0[6]), float(summary_o0[7]), float(summary_o0[8])],
    'O1': [float(summary_o1[6]), float(summary_o1[7]), float(summary_o1[8])],
    'O2': [float(summary_o2[6]), float(summary_o2[7]), float(summary_o2[8])]
}

df_speedup = pd.DataFrame(data, index=speedup_types)

# 使用seaborn创建更高级的柱状图
ax = df_speedup.plot(kind='bar', width=0.7, figsize=(14, 8), edgecolor='black')

# 添加基准线
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='基准线 (串行性能)')

# 添加折线图表示趋势
for i, col in enumerate(['O0', 'O1', 'O2']):
    plt.plot(range(len(speedup_types)), df_speedup[col], 
             marker=markers[i], markersize=12, linewidth=3, 
             color=colors[i], alpha=0.7)

plt.xlabel('SIMD实现类型')
plt.ylabel('加速比 (相对于串行实现)')
plt.title('不同编译优化级别下SIMD实现的加速比对比')
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加数据标签
for i, col in enumerate(['O0', 'O1', 'O2']):
    for j, val in enumerate(df_speedup[col]):
        plt.annotate(f'{val:.2f}x', 
                    xy=(j-0.3+i*0.3, val),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('advanced_speedup_comparison.png', dpi=300)
plt.close()

# 创建图3：热图展示优化级别与SIMD路数的交互影响
plt.figure(figsize=(14, 10))

# 准备热图数据 - 相对于基准(O0串行)的速度提升
base_speed = speed_o0_values[0]  # O0串行作为基准

data = np.array([
    [speed_o0_values[0]/base_speed, speed_o0_values[1]/base_speed, speed_o0_values[2]/base_speed, speed_o0_values[3]/base_speed],
    [speed_o1_values[0]/base_speed, speed_o1_values[1]/base_speed, speed_o1_values[2]/base_speed, speed_o1_values[3]/base_speed],
    [speed_o2_values[0]/base_speed, speed_o2_values[1]/base_speed, speed_o2_values[2]/base_speed, speed_o2_values[3]/base_speed]
])

# 使用seaborn创建高级热图
ax = sns.heatmap(data, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=methods, yticklabels=opt_levels,
                cbar_kws={'label': '相对于O0串行实现的性能比'},
                linewidths=1, annot_kws={"size": 14})

plt.title('编译优化级别与SIMD路数交互对性能的影响', fontsize=18, pad=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('advanced_performance_heatmap.png', dpi=300)
plt.close()

# 创建图4：雷达图展示不同编译优化级别的综合性能
plt.figure(figsize=(12, 10))

# 准备雷达图数据
categories = methods
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

def normalize_data(data_list):
    max_val = max([max(data) for data in data_list])
    return [[(val/max_val)*5 for val in data] for data in data_list]

# 归一化处理性能数据以适应雷达图
norm_data = normalize_data([speed_o0_values, speed_o1_values, speed_o2_values])
norm_o0 = norm_data[0] + [norm_data[0][0]]  # 闭合数据
norm_o1 = norm_data[1] + [norm_data[1][0]]
norm_o2 = norm_data[2] + [norm_data[2][0]]
cat_with_closure = categories + [categories[0]]  # 闭合标签

ax = plt.subplot(111, polar=True)
ax.plot(angles, norm_o0, 'o-', linewidth=2, label=opt_levels[0], color=colors[0])
ax.plot(angles, norm_o1, 's-', linewidth=2, label=opt_levels[1], color=colors[1])
ax.plot(angles, norm_o2, 'd-', linewidth=2, label=opt_levels[2], color=colors[2])
ax.fill(angles, norm_o0, alpha=0.1, color=colors[0])
ax.fill(angles, norm_o1, alpha=0.1, color=colors[1])
ax.fill(angles, norm_o2, alpha=0.1, color=colors[2])

# 添加雷达图标签和刻度
plt.xticks(angles[:-1], cat_with_closure[:-1], fontsize=14)
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], color="grey", size=10)
plt.ylim(0, 5)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('不同编译优化级别下各实现方式的相对性能雷达图', fontsize=18, y=1.08)

plt.tight_layout()
plt.savefig('advanced_performance_radar.png', dpi=300)
plt.close()