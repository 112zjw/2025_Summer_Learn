# -*- coding: utf-8 -*-
"""
二维 PSO 演示：每 20 次迭代生成 3D 粒子分布图
目标函数：f(x,y) = (x·sin(x)·cos(2x) - 2x·sin(3x)) * (y·sin(y)·cos(2y) - 2y·sin(3y))
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  仅为了激活 3D

# ------------------ 目标函数 ------------------
def objective(pos):
    """
    输入：pos -> ndarray, shape=(..., 2)
    输出：函数值 -> ndarray, shape=(...,)
    """
    x, y = pos[..., 0], pos[..., 1]
    fx = x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x)
    fy = y * np.sin(y) * np.cos(2 * y) - 2 * y * np.sin(3 * y)
    return fx * fy


# ------------------ PSO 类 ------------------
class PSO:
    def __init__(self,
                 func,
                 dim: int,
                 pop_size: int = 500,
                 max_iter: int = 100,
                 w_max: float = 0.9,
                 w_min: float = 0.4,
                 c1: float = 0.5,
                 c2: float = 0.5,
                 x_bound=None,
                 v_bound=None):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

        # 默认边界
        self.x_bound = np.asarray(x_bound) if x_bound is not None else np.array([[0, 1]] * dim)
        self.v_bound = np.asarray(v_bound) if v_bound is not None else np.array([[-1, 1]] * dim)

        # 初始化粒子
        self.X = np.random.uniform(self.x_bound[:, 0], self.x_bound[:, 1], (pop_size, dim))
        self.V = np.random.uniform(self.v_bound[:, 0], self.v_bound[:, 1], (pop_size, dim))

        # 个体最优
        self.pbest_X = self.X.copy()
        self.pbest_Y = self.func(self.X)
        # 全局最优
        self.gbest_idx = np.argmin(self.pbest_Y)
        self.gbest_X = self.pbest_X[self.gbest_idx].copy()
        self.gbest_Y = self.pbest_Y[self.gbest_idx]

    def update(self, iter_t):
        # 线性衰减惯性权重
        w = self.w_max - (self.w_max - self.w_min) * iter_t / self.max_iter

        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.V = (w * self.V +
                  self.c1 * r1 * (self.pbest_X - self.X) +
                  self.c2 * r2 * (self.gbest_X - self.X))
        self.V = np.clip(self.V, self.v_bound[:, 0], self.v_bound[:, 1])

        self.X = self.X + self.V
        self.X = np.clip(self.X, self.x_bound[:, 0], self.x_bound[:, 1])

        # 计算当前适应度
        y = self.func(self.X)

        # 更新个体最优
        update_mask = y < self.pbest_Y
        self.pbest_X[update_mask] = self.X[update_mask]
        self.pbest_Y[update_mask] = y[update_mask]

        # 更新全局最优
        gbest_idx = np.argmin(self.pbest_Y)
        if self.pbest_Y[gbest_idx] < self.gbest_Y:
            self.gbest_X = self.pbest_X[gbest_idx]
            self.gbest_Y = self.pbest_Y[gbest_idx]

    def run(self):
        # 存储历次粒子位置（用于外部绘图）
        history_pos = {0: self.X.copy()}
        for iter_t in range(1, self.max_iter + 1):
            self.update(iter_t - 1)
            # 每 20 代保存一次
            if iter_t % 20 == 0:
                history_pos[iter_t] = self.X.copy()
        return history_pos, self.gbest_X, self.gbest_Y


# ------------------ 主程序 ------------------
if __name__ == '__main__':
    # 搜索空间 [0,20]x[0,20]
    x_bound = [[0, 20], [0, 20]]
    v_bound = [[-1.5, 1.5], [-1.5, 1.5]]

    pso = PSO(objective,
              dim=2,
              pop_size=500,
              max_iter=100,
              w_max=0.9,
              w_min=0.4,
              c1=0.5,
              c2=0.5,
              x_bound=x_bound,
              v_bound=v_bound)

    history_pos, best_pos, best_val = pso.run()
    print("最优位置：", best_pos)
    print("最优值：", best_val)

    # ------------------ 3D 可视化 ------------------
    for iter_num, pos in history_pos.items():
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 计算颜色映射
        z = objective(pos)
        sc = ax.scatter(pos[:, 0], pos[:, 1], z,
                        c=z, cmap='viridis', s=8)

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(f"Iteration {iter_num}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("f(X,Y)")
        fig.colorbar(sc, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()