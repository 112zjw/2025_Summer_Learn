import random
import json
import matplotlib.pyplot as plt
import numpy as np
from util import *  # 导入calc(), plot_ans()等工具函数

# ============ 修复中文乱码设置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块的问题
# =========================================

# 步骤1：加载城市坐标数据
with open(r"D:\2025_Summer_Learn\CUMCM\运筹规划类\启发式算法\points.json", "r") as f:
    txt = f.readline()
points = json.loads(txt)
N = len(points)  # 城市数量
print(f"加载了 {N} 个城市坐标")

# 添加缺失的距离计算函数
def calc_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# 步骤2：初始化算法参数
# ------------------------------------------
# 参数说明:
# alpha: 信息素重要程度因子 
# beta: 启发函数重要程度因子
# ants: 蚂蚁数量
# evap: 信息素挥发系数(0~1)
# Q: 信息素强度常数
# iter_max: 最大迭代次数
# ------------------------------------------
alpha = 1       # 信息素重要程度因子
beta = 2        # 启发函数重要程度因子 (提高启发信息的权重)
ants = 20       # 蚂蚁数量 (增加蚂蚁数量以提高搜索能力)
evap = 0.4      # 信息素挥发系数
Q = 100         # 信息素强度常数
iter_max = 200  # 最大迭代次数 (增加到200次)

# 步骤3：初始化启发信息矩阵和信息素矩阵
# ------------------------------------------
# eta: 启发信息矩阵，存储城市间距离的倒数（距离越小，启发信息越大）
# gamma: 信息素矩阵，初始化为0.2
# ------------------------------------------
eta = [[0 for _ in range(N)] for _ in range(N)]
gamma = [[0.2 for _ in range(N)] for _ in range(N)]

# 计算城市间距离并填充启发信息矩阵
for i in range(N):
    for j in range(i+1, N):
        dist = calc_distance(points[i], points[j])
        eta[i][j] = 1 / dist
        eta[j][i] = 1 / dist

# 步骤4：定义蚂蚁路径选择函数
def choose_path(start, eta, gamma, alpha, beta):
    """蚂蚁构建路径的函数"""
    visited = [False] * N            # 访问标记数组
    visited[start] = True            # 起点已访问
    path = [start]                   # 路径列表，起点为start
    
    # 逐步选择下一个城市
    for _ in range(N - 1):
        current = path[-1]           # 当前所在城市
        candidates = []              # 候选城市列表
        weights = []                 # 选择权重列表
        
        # 计算所有未访问城市的概率
        for city in range(N):
            if not visited[city]:
                # 计算选择概率 = (信息素^alpha) * (启发信息^beta)
                prob = (gamma[current][city] ** alpha) * (eta[current][city] ** beta)
                candidates.append(city)
                weights.append(prob)
        
        # 根据概率选择下一个城市
        if weights:  # 确保有权重值
            next_city = random.choices(candidates, weights=weights, k=1)[0]
            path.append(next_city)
            visited[next_city] = True
        else:
            # 如果没有候选城市，随机选择一个未访问城市
            unvisited = [i for i in range(N) if not visited[i]]
            if unvisited:
                next_city = random.choice(unvisited)
                path.append(next_city)
                visited[next_city] = True
    
    return path  # 返回完整路径

# 步骤5：初始化记录变量
# ------------------------------------------
# 记录变量说明:
# rec_iter_best: 记录每次迭代的最优解
# rec_global_best: 记录全局最优解的变化
# rec_avg_length: 记录每次迭代的平均路径长度
# rec_pheromone: 记录每次迭代的平均信息素浓度
# ------------------------------------------
rec_iter_best = []       # 每次迭代的最优路径长度
rec_global_best = []     # 全局最优路径长度
rec_avg_length = []      # 每次迭代的平均路径长度
rec_pheromone = []       # 每次迭代的平均信息素浓度

best_path = None         # 全局最优路径
best_dist = float('inf') # 全局最短距离

# 步骤6：蚁群算法主循环
print("开始蚁群算法优化...")
for iteration in range(iter_max):
    # 6.1 初始化本次迭代的变量
    delta_gamma = [[0 for _ in range(N)] for _ in range(N)]  # 信息素增量矩阵
    iteration_lengths = []                                   # 存储本次迭代所有蚂蚁的路径长度
    iteration_best_dist = float('inf')                       # 本次迭代的最优路径长度
    iteration_best_path = None                               # 本次迭代的最优路径
    
    # 6.2 每只蚂蚁构建路径
    for ant in range(ants):
        # 每只蚂蚁从不同城市出发，增加多样性
        start_city = ant % N
        path = choose_path(start_city, eta, gamma, alpha, beta)
        
        # 计算路径长度
        path_length = calc(path, points)
        iteration_lengths.append(path_length)
        
        # 更新本次迭代的最优解
        if path_length < iteration_best_dist:
            iteration_best_dist = path_length
            iteration_best_path = path
        
        # 更新全局最优解
        if path_length < best_dist:
            best_dist = path_length
            best_path = path.copy()
            print(f"迭代 {iteration}: 发现新的全局最优解 {best_dist:.2f}")
        
        # 6.3 更新信息素增量
        # 只有较优解释放信息素（精英策略）
        if path_length < 1.5 * best_dist:  # 只允许比当前最优解差50%以内的路径释放信息素
            for i in range(N - 1):
                city1, city2 = path[i], path[i+1]
                # 释放信息素：Q / 路径长度（路径越短，释放的信息素越多）
                delta = Q / path_length
                delta_gamma[city1][city2] += delta
                delta_gamma[city2][city1] += delta
            
            # 处理起点和终点
            first, last = path[0], path[-1]
            delta = Q / path_length
            delta_gamma[first][last] += delta
            delta_gamma[last][first] += delta
    
    # 6.4 更新信息素矩阵（先挥发，再增加）
    for i in range(N):
        for j in range(N):
            gamma[i][j] = gamma[i][j] * evap + delta_gamma[i][j]
    
    # 6.5 记录本次迭代数据
    rec_iter_best.append(iteration_best_dist)
    rec_global_best.append(best_dist)
    rec_avg_length.append(np.mean(iteration_lengths))
    
    # 计算平均信息素浓度
    pheromone_sum = 0
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            pheromone_sum += gamma[i][j]
            count += 1
    if count > 0:
        rec_pheromone.append(pheromone_sum / count)
    else:
        rec_pheromone.append(0)
    
    # 6.6 每20次迭代打印进度
    if iteration % 20 == 0 or iteration == iter_max - 1:
        print(f"迭代: {iteration:3d}/{iter_max}, 本次最优: {iteration_best_dist:.2f}, 全局最优: {best_dist:.2f}, 平均长度: {np.mean(iteration_lengths):.2f}")

print("\n优化完成!")
print(f"最优路径长度: {best_dist:.2f}")

# 步骤7：可视化结果
# 7.1 绘制最优路径图
plt.figure(figsize=(10, 8))
plot_ans(best_path, points)
plt.title(f"蚁群算法最优路径 (长度: {best_dist:.2f})")
plt.tight_layout()

# 7.2 创建多图展示收敛过程
plt.figure(figsize=(14, 10))

# 7.2.1 路径长度收敛曲线 (左上)
plt.subplot(2, 2, 1)
plt.plot(rec_iter_best, 'b-', linewidth=1.5, alpha=0.7, label='本次迭代最优')
plt.plot(rec_global_best, 'r-', linewidth=2, label='全局最优')
plt.plot(rec_avg_length, 'g--', linewidth=1.5, alpha=0.7, label='平均路径长度')

# 添加移动平均线（展示趋势）
window_size = 10
if len(rec_avg_length) >= window_size:
    moving_avg = np.convolve(rec_avg_length, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rec_avg_length)), moving_avg, 'm-', linewidth=2, label=f'{window_size}次移动平均')

plt.xlabel('迭代次数')
plt.ylabel('路径长度')
plt.title('路径长度收敛曲线')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 添加标注
if rec_iter_best:
    plt.annotate('快速收敛阶段', 
                 xy=(20, rec_iter_best[20] if len(rec_iter_best) > 20 else rec_iter_best[0]), 
                 xytext=(40, (rec_iter_best[20] if len(rec_iter_best) > 20 else rec_iter_best[0]) + 20),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.5))

    plt.annotate('精细调整阶段', 
                 xy=(150, rec_global_best[-1] if len(rec_global_best) > 150 else rec_global_best[-1]), 
                 xytext=(100, rec_global_best[-1] + 20),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5))

# 7.2.2 信息素浓度变化曲线 (右上)
plt.subplot(2, 2, 2)
if rec_pheromone:
    plt.plot(rec_pheromone, 'm-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('平均信息素浓度')
    plt.title('信息素浓度变化')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加标注
    plt.annotate('信息素积累', 
                 xy=(50, rec_pheromone[50] if len(rec_pheromone) > 50 else rec_pheromone[0]), 
                 xytext=(80, (rec_pheromone[50] if len(rec_pheromone) > 50 else rec_pheromone[0]) + 0.01),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)

    plt.annotate('稳定平衡', 
                 xy=(150, rec_pheromone[-1]), 
                 xytext=(100, rec_pheromone[-1] + 0.01),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)

# 7.2.3 信息素矩阵可视化 (左下)
plt.subplot(2, 2, 3)
# 只显示前20个城市的信息素矩阵，避免过多数据
display_size = min(20, N)
display_gamma = np.array(gamma)[:display_size, :display_size]

plt.imshow(display_gamma, cmap='hot', interpolation='nearest')
plt.colorbar(label='信息素浓度')
plt.xlabel('城市索引')
plt.ylabel('城市索引')
plt.title(f'前{display_size}个城市的信息素矩阵')
plt.xticks(range(display_size))
plt.yticks(range(display_size))

# 7.2.4 路径长度对比 (右下)
plt.subplot(2, 2, 4)
if rec_iter_best:
    iterations = list(range(len(rec_iter_best)))
    bar_width = 0.35

    plt.bar(iterations, rec_avg_length, width=bar_width, alpha=0.6, label='平均长度')
    plt.bar([i + bar_width for i in iterations], rec_iter_best, width=bar_width, alpha=0.8, label='迭代最优')
    plt.plot(rec_global_best, 'r-', linewidth=2, label='全局最优')

    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('路径长度对比')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()
plt.show()

# 步骤8：输出最终结果
print("\n最优路径:", best_path)
print(f"最短距离: {best_dist:.2f}")