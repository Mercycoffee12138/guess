import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

# 创建图形1：不同优化级别的总体性能比较（每秒处理密码数）
plt.figure(figsize=(12, 6))
methods = ['串行', '二路SIMD', '四路SIMD', '八路SIMD']
speed_o0_values = [float(speed_o0[2]), float(speed_o0[3]), float(speed_o0[4]), float(speed_o0[5])]
speed_o1_values = [float(speed_o1[2]), float(speed_o1[3]), float(speed_o1[4]), float(speed_o1[5])]
speed_o2_values = [float(speed_o2[2]), float(speed_o2[3]), float(speed_o2[4]), float(speed_o2[5])]

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, speed_o0_values, width, label='O0 (无优化)')
plt.bar(x, speed_o1_values, width, label='O1 (一级优化)')
plt.bar(x + width, speed_o2_values, width, label='O2 (二级优化)')

plt.xlabel('实现方式')
plt.ylabel('每秒处理密码数')
plt.title('不同优化级别下各实现的处理速度比较')
plt.xticks(x, methods)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 格式化y轴标签为百万级
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig('processing_speed_comparison.png', dpi=300)
plt.close()

# 创建图形2：不同优化级别的加速比比较
plt.figure(figsize=(12, 6))
speedup_types = ['二路加速比', '四路加速比', '八路加速比']
speedup_o0 = [float(summary_o0[6]), float(summary_o0[7]), float(summary_o0[8])]
speedup_o1 = [float(summary_o1[6]), float(summary_o1[7]), float(summary_o1[8])]
speedup_o2 = [float(summary_o2[6]), float(summary_o2[7]), float(summary_o2[8])]

x = np.arange(len(speedup_types))
width = 0.25

plt.bar(x - width, speedup_o0, width, label='O0 (无优化)')
plt.bar(x, speedup_o1, width, label='O1 (一级优化)')
plt.bar(x + width, speedup_o2, width, label='O2 (二级优化)')

plt.axhline(y=1, color='r', linestyle='--', label='基准线 (串行性能)')

plt.xlabel('SIMD实现')
plt.ylabel('加速比 (相对于串行实现)')
plt.title('不同优化级别下SIMD实现的加速比')
plt.xticks(x, speedup_types)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300)
plt.close()

# 创建图形3：每批次的八路SIMD加速比趋势
plt.figure(figsize=(12, 6))
plt.plot(df_o0['批次'], df_o0['八路加速比'], 'o-', label='O0 (无优化)')
plt.plot(df_o1['批次'], df_o1['八路加速比'], 's-', label='O1 (一级优化)')
plt.plot(df_o2['批次'], df_o2['八路加速比'], '^-', label='O2 (二级优化)')

plt.axhline(y=1, color='r', linestyle='--', label='基准线 (串行性能)')

plt.xlabel('批次')
plt.ylabel('八路SIMD加速比')
plt.title('不同批次下八路SIMD的加速比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('batch_speedup_trend.png', dpi=300)
plt.close()

# 创建图形4：优化级别对总处理时间的影响
plt.figure(figsize=(12, 6))
methods = ['串行', '二路SIMD', '四路SIMD', '八路SIMD']
time_o0 = [float(summary_o0[2]), float(summary_o0[3]), float(summary_o0[4]), float(summary_o0[5])]
time_o1 = [float(summary_o1[2]), float(summary_o1[3]), float(summary_o1[4]), float(summary_o1[5])]
time_o2 = [float(summary_o2[2]), float(summary_o2[3]), float(summary_o2[4]), float(summary_o2[5])]

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, time_o0, width, label='O0 (无优化)')
plt.bar(x, time_o1, width, label='O1 (一级优化)')
plt.bar(x + width, time_o2, width, label='O2 (二级优化)')

plt.xlabel('实现方式')
plt.ylabel('总处理时间 (秒)')
plt.title('不同优化级别下各实现的总处理时间')
plt.xticks(x, methods)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('total_processing_time.png', dpi=300)
plt.close()

# 创建图形5：热力图显示不同优化级别的相对性能
plt.figure(figsize=(10, 8))
data = np.array([
    [speed_o0_values[0]/speed_o0_values[0], speed_o0_values[1]/speed_o0_values[0], 
     speed_o0_values[2]/speed_o0_values[0], speed_o0_values[3]/speed_o0_values[0]],
    [speed_o1_values[0]/speed_o0_values[0], speed_o1_values[1]/speed_o0_values[0], 
     speed_o1_values[2]/speed_o0_values[0], speed_o1_values[3]/speed_o0_values[0]],
    [speed_o2_values[0]/speed_o0_values[0], speed_o2_values[1]/speed_o0_values[0], 
     speed_o2_values[2]/speed_o0_values[0], speed_o2_values[3]/speed_o0_values[0]]
])

sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu',
            xticklabels=methods, yticklabels=['O0', 'O1', 'O2'],
            cbar_kws={'label': '相对于O0串行实现的性能比'})

plt.title('不同优化级别和SIMD实现的相对性能热力图')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300)
plt.close()