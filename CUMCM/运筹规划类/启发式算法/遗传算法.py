import random
from util import *  # 假设包含calc(), plot_ans()等函数
import matplotlib.pyplot as plt
import json

# 读取城市坐标数据
with open(r"D:\2025_Summer_Learn\CUMCM\运筹规划类\启发式算法\points.json", "r") as f:
    txt = f.readline()
points = json.loads(txt)
N = len(points)  # 城市数量

# 遗传算法参数
M = 10         # 种群大小
p_cross = 1    # 交叉概率
p_perm = 0.1   # 变异概率

# 选择函数：评估种群并选择最优的M个个体
def select(population, M, points):
    ranking = []
    for p in population:
        ranking.append((calc(p, points), p))  # 计算路径长度
    ranking.sort(key=lambda x: x[0])  # 按路径长度排序
    return [t[1] for t in ranking[:M]]  # 返回前M个最优个体

# 变异操作：2-opt局部优化（随机反转一段路径）
def perm(route):
    i = random.randint(1, N-2)  # 避免端点
    j = random.randint(1, N-2)
    if i > j:
        i, j = j, i
    # 反转i到j的子路径
    return route[:i] + list(reversed(route[i:j+1])) + route[j+1:]

# 交叉操作：顺序交叉OX
def cross(r1, r2):
    i = random.randint(1, N-2)
    j = random.randint(1, N-2)
    if i > j:
        i, j = j, i
    
    # 从r1截取i-j的子路径
    segment = r1[i:j+1]
    
    # 构建子代1：保留r1的segment，按r2顺序插入剩余城市
    child1 = [city for city in r2 if city not in segment]
    child1 = child1[:i] + segment + child1[i:]
    
    # 构建子代2：保留r2的segment，按r1顺序插入剩余城市
    child2 = [city for city in r1 if city not in segment]
    child2 = child2[:i] + segment + child2[i:]
    
    return child1, child2

# 初始化种群
population = []
for _ in range(M * 10):  # 初始种群规模为10M
    path = list(range(N))
    random.shuffle(path)
    population.append(path)
population = select(population, M, points)  # 选择最优的M个个体

rec = []  # 记录每代最优路径长度

# 遗传算法主循环
for iter in range(100):  # 增加迭代次数至100代
    new_gen = population.copy()  # 直接复制当前种群
    
    # 交叉操作：遍历所有个体对
    for i in range(M):
        for j in range(i+1, M):  # 避免自交和重复配对
            if random.random() < p_cross:
                child1, child2 = cross(population[i], population[j])
                new_gen.extend([child1, child2])
    
    # 变异操作：遍历新种群中的每个个体
    for idx in range(len(new_gen)):
        if random.random() < p_perm:
            new_gen[idx] = perm(new_gen[idx])  # 直接修改列表元素
    
    # 选择下一代
    population = select(new_gen, M, points)
    best_length = calc(population[0], points)
    rec.append(best_length)  # 记录当代最优解
    print(f"Gen {iter}: Best = {best_length}")

# 输出结果
best_path = population[0]
print("最优路径:", best_path)
plot_ans(best_path, points)  # 绘制路径图

# 绘制收敛曲线
plt.figure()
plt.plot(rec)
plt.xlabel('Generation')
plt.ylabel('Best Path Length')
plt.title('Genetic Algorithm Convergence')
plt.grid(True)
plt.show()