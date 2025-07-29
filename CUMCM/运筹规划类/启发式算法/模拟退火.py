import random
import math
import json
import matplotlib.pyplot as plt
import numpy as np
from util import *  # 导入自定义工具函数(greedy, calc, plot_ans)

# ============ 修复中文乱码设置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块的问题
# =========================================

# 步骤1：加载城市坐标数据
with open(r"D:\2025_Summer_Learn\CUMCM\运筹规划类\启发式算法\points.json", "r") as f:
    txt = f.readline()
points = json.loads(txt)
N = len(points)

# 步骤2：生成初始解（使用贪婪算法）
last = greedy(0, points)

def perm(ans):
    """邻域扰动函数（2-opt操作）"""
    i = random.randint(1, N-1)
    j = random.randint(1, N-1)
    if i > j:
        i, j = j, i
    new_ans = ans[:i] + ans[j:i-1:-1] + ans[j+1:]
    return new_ans

# 创建三个记录列表[1,6](@ref)
rec_current = []  # 记录每次迭代的当前解路径长度（展示波动）
rec_best = []     # 记录每次迭代的全局最优路径长度（展示收敛趋势）
rec_temperature = []  # 记录温度变化

# 步骤3：初始化模拟退火参数
T = 20         # 初始温度
alpha = 0.999  # 温度衰减系数
cur_l = calc(last, points)
best = last.copy()
best_length = cur_l  # 全局最优解的路径长度

# 步骤4：模拟退火主循环
for iter in range(10000):
    # 4.1 生成邻域解
    tmp = perm(last)
    new_l = calc(tmp, points)
    
    # 4.2 Metropolis准则判断是否接受新解
    if new_l < cur_l:  # 新解更优：直接接受
        last = tmp
        cur_l = new_l
        # 更新全局最优解
        if new_l < best_length:
            best = tmp.copy()
            best_length = new_l
    else:  # 新解更差：以概率接受
        delta = new_l - cur_l
        accept_prob = math.exp(-delta / T)
        if random.random() < accept_prob:
            last = tmp
            cur_l = new_l
    
    # 4.3 降温操作
    T *= alpha
    
    # 4.4 记录数据[2,5](@ref)
    rec_current.append(cur_l)  # 当前解路径长度（会有波动）
    rec_best.append(best_length)  # 历史最优解路径长度
    rec_temperature.append(T)  # 温度变化
    
    # 4.5 每1000次迭代打印进度
    if iter % 1000 == 0:
        print(f"迭代: {iter}, 温度: {T:.4f}, 当前长度: {cur_l:.4f}, 最优长度: {best_length:.4f}")

# 步骤5：输出最终结果
print("最优路径:", best)
print("最短距离:", best_length)

# 步骤6：可视化结果
# 6.1 绘制路径图
plot_ans(best, points)

# 6.2 绘制收敛曲线（双Y轴图表）
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左侧Y轴：路径长度
color = 'tab:blue'
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('路径长度', color=color)
ax1.plot(rec_current, 'b-', linewidth=0.8, alpha=0.7, label='当前解')
ax1.plot(rec_best, 'r-', linewidth=1.5, label='历史最优')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

# 右侧Y轴：温度变化[4](@ref)
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('温度', color=color)
ax2.plot(rec_temperature, 'g-', linewidth=1, alpha=0.6, label='温度')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')  # 对数坐标更清晰展示温度衰减

# 添加图例和标题
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("模拟退火算法收敛过程 (双Y轴图表)")
fig.tight_layout()

# 6.3 添加波动区域标注[3](@ref)
ax1.annotate('高波动区域\n(高温期接受劣解)', 
             xy=(200, max(rec_current[:500])), 
             xytext=(300, max(rec_current[:500]) + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

ax1.annotate('收敛区域\n(低温期只接受优解)', 
             xy=(8000, rec_best[-1]), 
             xytext=(6000, rec_best[-1] + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3))

# 6.4 添加移动平均线（展示趋势）
window_size = 100  # 100次迭代的移动平均
moving_avg = np.convolve(rec_current, np.ones(window_size)/window_size, mode='valid')
ax1.plot(range(window_size-1, len(rec_current)), moving_avg, 'm--', linewidth=2, label=f'{window_size}次移动平均')

plt.show()