import numpy as np

# data1 = np.load("output_decoding_times_speculative_20250225_182212.npy")
# data2 = np.load("output_decoding_times_speculative.npy")

# for i in range(len(data1)):
#     print((data1[i]-data2[i])/data1[i])

history = [0,0.0000,0.6000,0.6000,0.0000,0.4000,0.4000,1.0000,1.0000,1.0000,0.6000,0.4000,0.8000,1.0000,0.4000,0.8000,0.4000,0.4000,0.8000,0.8000,0.6000,0.6000,0.6000,0.6000,0.8000,1.0000,0.6000,0.8000,1.0000,1.0000,0.8000,0.6000,0.6000,1.0000,0.6000,0.6000,1.0000,1.0000,0.4000,0.4000,0.4000,0.6000,0.8000,0.6000,0.6000,0.8000,0.8000,0.6000,1.0000,1.0000,1.0000,1.0000,0.8000,0.6000,0.8000,0.6000,0.6000,1.0000,0.8000,0.2000,0.6000,0.8000,0.6000,0.4000,0.8000,0.4000,0.6000,0.4000,0.8000,0.6000,0.6000,0.6000,0.6000,0.6000,0.6000,0.6000,0.4000,0.6000,0.4000,0.4000,0.2000,0.6000,1.0000,1.0000,0.6000,0.8000,1.0000,0.6000,0.6000,0.8000,0.6000,0.8000,0.8000,0.8000,0.8000,0.6000,0.4000,0.0000,0.0000,0.4000,0.0000,0.6000,0.2000,0.8000,0.6000,0.8000,0.4000,0.4000,0.6000,0.6000,0.0000,0.4000,0.4000,1.0000,1.0000,1.0000,0.6000,0.4000,0.8000,1.0000,0.4000,0.8000,0.4000,0.4000,0.8000,0.8000,0.6000,0.6000,0.6000,0.6000,0.8000,1.0000,0.6000,0.8000,1.0000,1.0000,0.8000,0.6000,0.6000,1.0000,0.6000,0.6000,1.0000,1.0000,0.4000,0.4000,0.4000,0.6000,0.8000,0.6000,0.6000,0.8000,0.8000,0.6000,1.0000,1.0000,1.0000,1.0000,0.8000,0.6000,0.8000,0.6000,0.6000,1.0000,0.8000,0.2000,0.6000,0.8000,0.6000,0.4000,0.8000,0.4000,0.6000,0.4000,0.8000,0.6000,0.6000,0.6000,0.6000,0.6000,0.6000,0.6000,0.4000,0.6000,0.4000,0.4000,0.2000,0.6000,1.0000,1.0000,0.6000,0.8000,1.0000,0.6000,0.6000,0.8000,0.6000,0.8000,0.8000,0.8000,0.8000,0.6000,0.4000,0.0000,0.0000,0.4000,0.0000,0.6000,0.2000,0.8000,0.6000,0.8000]
new = [int(i*5) for i in history]

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用 Noto Sans CJK
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 假设这是你的一维数据
data = new # [5, 7, 8, 6, 9, 4, 3, 10]
plt.figure(figsize=(10, 5))
# 生成 X 轴（数据的顺序）
x = range(len(data))

# Create a color list where only bars with value 0 are visible
colors = ['red' if val == 0 else 'blue' for val in data]
plt.bar(x, data)

# 绘制柱状图
#plt.bar(x, data)

# 添加标题和标签
plt.title('一维数据的柱状图')
plt.xlabel('数据顺序')
plt.ylabel('数据值')

plt.savefig('bar_chart.png') 
# Calculate CDF
sorted_data = np.sort(data)
cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# Create CDF plot
plt.figure(figsize=(10, 5))
plt.plot(sorted_data, cumulative_prob, marker='.', linestyle='-')
plt.grid(True)

plt.title('累积分布函数 (CDF)')
plt.xlabel('数据值')
plt.ylabel('累积概率')

plt.savefig('cdf_chart.png')

# Create density plot using kernel density estimation
plt.figure(figsize=(10, 5))
density = plt.hist(data, bins=6, density=True, alpha=0.7)
plt.grid(True)

plt.title('密度分布图')
plt.xlabel('数据值')
plt.ylabel('密度')

plt.savefig('density_chart.png')

