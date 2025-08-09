"""
遗传算法（Genetic Algorithm, GA）演示
问题：在区间 [-10, 10] 内找 f(x)=x^2 的最大值
"""

import random
import math

# ========== 1. 基本参数 ==========
POP_SIZE       = 40          # 种群大小
CHROM_LEN      = 16          # 染色体长度（二进制位数）
CROSS_RATE     = 0.8         # 交叉概率
MUTATE_RATE    = 0.01        # 变异概率
GENERATIONS    = 100         # 演化代数
X_BOUND        = (-10, 10)   # 变量 x 的取值范围

# ========== 2. 工具函数 ==========
def binary2real(binary, bound):
    """
    将二进制串转化为区间 bound 内的实数
    binary: list[int]，例如 [1,0,1,0,...]
    bound : (low, high)
    """
    # 二进制转十进制
    dec = int(''.join(map(str, binary)), 2)
    # 线性映射到指定区间
    low, high = bound
    max_dec = 2 ** len(binary) - 1
    return low + dec * (high - low) / max_dec

def real2binary(x, bound, length):
    """
    将实数 x 编码为二进制串（仅作演示，GA 运行时不一定用到）
    """
    low, high = bound
    max_dec = 2 ** length - 1
    dec = int((x - low) / (high - low) * max_dec)
    binary = bin(dec)[2:].zfill(length)
    return list(map(int, binary))

# ========== 3. 目标函数 ==========
def fitness_func(x):
    """目标函数：f(x)=x^2，需要最大化"""
    return x ** 2

# ========== 4. 初始化种群 ==========
def init_population(pop_size, chrom_len):
    """
    返回一个种群：list[list[int]]，每个个体是 0/1 列表
    """
    return [[random.randint(0, 1) for _ in range(chrom_len)]
            for _ in range(pop_size)]

# ========== 5. 选择操作（锦标赛选择） ==========
def tournament_select(pop, k=3):
    """从种群中随机取 k 个个体，返回适应度最高的那个"""
    candidates = random.sample(pop, k)
    best = max(candidates, key=lambda ind: fitness_func(binary2real(ind, X_BOUND)))
    return best

# ========== 6. 交叉操作（单点交叉） ==========
def crossover(parent1, parent2, cross_rate):
    """以 cross_rate 的概率进行交叉，返回两个子代"""
    if random.random() > cross_rate:
        return parent1[:], parent2[:]   # 不交叉，直接复制
    # 随机交叉点
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# ========== 7. 变异操作（位翻转变异） ==========
def mutate(ind, mutate_rate):
    """以 mutate_rate 的概率翻转每一位"""
    for i in range(len(ind)):
        if random.random() < mutate_rate:
            ind[i] ^= 1  # 0 变 1，1 变 0
    return ind

# ========== 8. 主循环 ==========
def genetic_algorithm():
    pop = init_population(POP_SIZE, CHROM_LEN)

    for gen in range(GENERATIONS):
        # 计算当前种群适应度
        fitness = [fitness_func(binary2real(ind, X_BOUND)) for ind in pop]
        best_x  = binary2real(pop[fitness.index(max(fitness))], X_BOUND)
        best_f  = max(fitness)

        # 打印当前代信息
        print(f"第 {gen:>3} 代 | 最优 x = {best_x:>8.4f} | f(x) = {best_f:>8.2f}")

        # 生成新一代种群
        new_pop = []
        while len(new_pop) < POP_SIZE:
            # 选择
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)
            # 交叉
            c1, c2 = crossover(p1, p2, CROSS_RATE)
            # 变异
            c1 = mutate(c1, MUTATE_RATE)
            c2 = mutate(c2, MUTATE_RATE)
            new_pop.extend([c1, c2])
        pop = new_pop[:POP_SIZE]  # 保持种群大小不变

    # 最终输出
    final_fitness = [fitness_func(binary2real(ind, X_BOUND)) for ind in pop]
    best_idx      = final_fitness.index(max(final_fitness))
    best_ind      = pop[best_idx]
    best_x        = binary2real(best_ind, X_BOUND)
    best_f        = max(final_fitness)
    print("\n==== 结果 ====")
    print(f"最优个体: {''.join(map(str, best_ind))}")
    print(f"最优 x  : {best_x}")
    print(f"最优 f  : {best_f}")

if __name__ == "__main__":
    genetic_algorithm()