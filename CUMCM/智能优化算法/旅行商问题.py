# -*- coding: utf-8 -*-
"""
旅行商问题（TSP）——遗传算法求解 + 可视化
运行：python tsp_ga.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

# ------------------ 城市 ------------------
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.hypot(self.x - city.x, self.y - city.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"

# ------------------ 工具函数 ------------------
def create_cities(n_cities=30, area=100):
    """随机生成 n_cities 个城市的坐标"""
    return [City(np.random.uniform(0, area), np.random.uniform(0, area)) for _ in range(n_cities)]

def total_distance(route):
    """计算一条回路总距离"""
    dist = 0
    for i in range(len(route)):
        dist += route[i].distance(route[(i + 1) % len(route)])
    return dist

# ------------------ 遗传算法 ------------------
class GA:
    def __init__(self, cities, pop_size=200, elite_size=20, mutation_rate=0.01, generations=500):
        self.cities = cities
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def create_individual(self):
        """随机排列城市 -> 个体"""
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def rank_routes(self, population):
        """计算并排序所有个体的适应度（距离越小越优）"""
        fitness = [(total_distance(ind), ind) for ind in population]
        return sorted(fitness, key=lambda x: x[0])

    def selection(self, ranked_pop):
        """锦标赛选择"""
        selected = []
        for _ in range(self.elite_size):
            selected.append(ranked_pop[_][1])
        for _ in range(len(ranked_pop) - self.elite_size):
            a, b = random.sample(ranked_pop[:50], 2)  # 从前50里选2个
            selected.append(a[1] if a[0] < b[0] else b[1])
        return selected

    def crossover(self, parent1, parent2):
        """有序交叉 OX"""
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child_p1 = parent1[a:b]
        child_p2 = [item for item in parent2 if item not in child_p1]
        return child_p2[:a] + child_p1 + child_p2[a:]

    def mutate(self, individual):
        """交换变异"""
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * len(individual))
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual

    def next_generation(self, current_gen):
        ranked = self.rank_routes(current_gen)
        selection_results = self.selection(ranked)
        children = selection_results[:self.elite_size]
        pool = random.sample(selection_results, len(selection_results))
        for i in range(len(selection_results) - self.elite_size):
            child = self.crossover(pool[i], pool[len(selection_results) - i - 1])
            children.append(self.mutate(child))
        return children

    def run(self):
        pop = self.initial_population()
        progress = []
        for gen in range(self.generations):
            pop = self.next_generation(pop)
            best_distance = total_distance(pop[0])
            progress.append(best_distance)
            if gen % 50 == 0 or gen == self.generations - 1:
                print(f"Gen {gen:3d}: best distance = {best_distance:.2f}")
        best_route = self.rank_routes(pop)[0][1]
        return best_route, progress

# ------------------ 可视化 ------------------
def plot_result(cities, best_route, progress):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 收敛曲线
    ax1.plot(progress)
    ax1.set_title("Convergence Curve")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Distance")

    # 路线图
    x = [c.x for c in best_route] + [best_route[0].x]
    y = [c.y for c in best_route] + [best_route[0].y]
    ax2.plot(x, y, 'o-')
    ax2.set_title("Best Route")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    for c in best_route:
        ax2.annotate(f"{c.x:.0f},{c.y:.0f}", (c.x, c.y), fontsize=8)
    plt.tight_layout()
    plt.show()

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    cities = create_cities(n_cities=30, area=100)
    ga = GA(cities, pop_size=300, elite_size=30, mutation_rate=0.02, generations=400)
    best_route, progress = ga.run()
    plot_result(cities, best_route, progress)