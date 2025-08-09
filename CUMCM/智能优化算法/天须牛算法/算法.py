#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版天牛群搜索算法（Enhanced BAS）完整实现
-------------------------------------------------
核心改进：
1. 种群机制替代单个体（解决全局搜索能力弱问题）[5,6](@ref)
2. 非线性自适应步长衰减（避免固定衰减导致的早熟收敛）[7](@ref)
3. Lévy Flight扰动策略（增强跳出局部最优能力）[6](@ref)
4. 精英保留与混合选择机制（平衡探索与开发）[5](@ref)

引用论文：
[1] Ye et al., "MO-BSAS: Multi-operator Beetle Swarm Antennae Search", Soft Computing (2024)
[2] 马吉明 等, "基于混沌扰动机制的天牛须搜索算法", 轻工学报 (2019)
[3] 扶笃雄 等, "基于天牛须种群算法的整周模糊度解算方法", CN116908899A (2023)

作者：AI助手
日期：2025-08-08
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import special

# ------------------------------------------------------------------
# 1. 目标函数：Ackley 函数（2 维）
# ------------------------------------------------------------------
def ackley_2d(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """
    Ackley 函数（2维）- 多峰测试函数
    全局最小值：f(0,0) = 0
    """
    x1, x2 = x[0], x[1]
    term1 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2))
    term2 = -np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2)
    return term1 + term2 + a + np.e

# ------------------------------------------------------------------
# 2. 改进版天牛群算法（Enhanced BAS）
# ------------------------------------------------------------------
class EnhancedBAS:
    """
    改进点：
    1. 种群机制（非单个体）
    2. 非线性自适应步长
    3. Lévy Flight扰动
    4. 精英保留策略
    """
    
    def __init__(
        self,
        func,
        n_dim: int,
        bounds: tuple = (-5, 5),
        pop_size: int = 30,          # 种群大小 [5,6](@ref)
        step_max: float = 2.0,        # 最大步长
        step_min: float = 0.01,        # 最小步长
        d0: float = 3.0,              # 初始触须间距
        max_iter: int = 200,
        tol: float = 1e-6,
        levy_scale: float = 0.1,      # Lévy Flight缩放因子 [6](@ref)
        seed: int = None
    ):
        self.func = func
        self.n_dim = n_dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.step_max = step_max
        self.step_min = step_min
        self.d0 = d0
        self.max_iter = max_iter
        self.tol = tol
        self.levy_scale = levy_scale
        self.rng = np.random.default_rng(seed)
        
        # 记录优化过程
        self.history = {'f_best': [], 'x_best': [], 'positions': []}
        self.x_best = None
        self.f_best = np.inf

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """裁剪到边界内"""
        low, high = self.bounds
        return np.clip(x, low, high)

    def _levy_flight(self, size: tuple) -> np.ndarray:
        """生成Lévy Flight扰动向量"""
        beta = 1.5
        # 使用scipy.special.gamma替代np.emath.gamma
        sigma = (special.gamma(1+beta) * np.sin(np.pi*beta/2) / 
                 (special.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = self.rng.normal(0, sigma, size)
        v = self.rng.normal(0, 1, size)
        return self.levy_scale * u / np.abs(v)**(1/beta)

    def _update_step(self, iter: int) -> float:
        """非线性自适应步长更新 [7](@ref)"""
        return self.step_min + (self.step_max - self.step_min) * (1 - (iter / self.max_iter)**0.5)

    def _update_d0(self, iter: int) -> float:
        """触须间距衰减"""
        return self.d0 * (1 - iter / self.max_iter)

    def run(self) -> tuple:
        """执行优化主循环"""
        low, high = self.bounds
        # 1. 初始化天牛种群 [5](@ref)
        population = self.rng.uniform(low, high, size=(self.pop_size, self.n_dim))
        fitness = np.apply_along_axis(self.func, 1, population)
        
        # 记录全局最优
        best_idx = np.argmin(fitness)
        self.x_best = population[best_idx].copy()
        self.f_best = fitness[best_idx]
        
        # 记录历史
        self.history['f_best'].append(self.f_best)
        self.history['x_best'].append(self.x_best.copy())
        self.history['positions'].append(population.copy())
        
        # 2. 主迭代循环
        for it in range(self.max_iter):
            # 更新步长和触须间距
            step = self._update_step(it)
            d_current = self._update_d0(it)
            
            # 3. 更新每个天牛位置
            for i in range(self.pop_size):
                # 3.1 生成随机方向并归一化
                dir_vec = self.rng.standard_normal(self.n_dim)
                dir_vec /= np.linalg.norm(dir_vec) + 1e-8
                
                # 3.2 计算左右触须位置
                x_left = self._clip(population[i] + dir_vec * d_current / 2)
                x_right = self._clip(population[i] - dir_vec * d_current / 2)
                
                # 3.3 评估触须位置
                f_left = self.func(x_left)
                f_right = self.func(x_right)
                
                # 3.4 更新位置
                population[i] = self._clip(
                    population[i] - step * dir_vec * np.sign(f_left - f_right)
                )
            
            # 4. 应用Lévy Flight扰动到最优个体[6](@ref)
            levy_move = self._levy_flight(self.n_dim)
            candidate = self._clip(self.x_best + levy_move)
            f_candidate = self.func(candidate)
            
            # 5. 混合选择：精英保留+扰动接受
            new_fitness = np.apply_along_axis(self.func, 1, population)
            
            # 5.1 更新种群最优
            min_idx = np.argmin(new_fitness)
            min_fitness = new_fitness[min_idx]
            
            # 5.2 接受更优的扰动解
            if f_candidate < min_fitness:
                population[min_idx] = candidate  # 替换最差个体
                min_fitness = f_candidate
                min_candidate = candidate
            else:
                min_candidate = population[min_idx]
            
            # 6. 更新全局最优
            if min_fitness < self.f_best:
                self.x_best = min_candidate.copy()
                self.f_best = min_fitness
            
            # 记录历史
            self.history['f_best'].append(self.f_best)
            self.history['x_best'].append(self.x_best.copy())
            self.history['positions'].append(population.copy())
            
            # 7. 收敛检测（多条件）[4](@ref)
            if it > 10:
                # 检测最近10次迭代的改进
                recent_improve = abs(self.history['f_best'][-10] - self.f_best)
                if recent_improve < self.tol and step < 0.05:
                    print(f"[CONVERGED] Iteration {it+1}, f_best={self.f_best:.10f}")
                    break

        return self.x_best, self.f_best

    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['f_best'], 'o-', color='royalblue', markersize=4, label='Best Fitness')
        plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("f(x)")
        plt.title("Enhanced BAS Convergence Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def plot_search_path_3d(self):
        """三维可视化搜索轨迹（仅支持2维问题）"""
        if self.n_dim != 2:
            print("Warning: 3D visualization only for 2D problems")
            return
            
        # 生成函数曲面
        low, high = self.bounds
        x = np.linspace(low, high, 100)
        y = np.linspace(low, high, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制函数曲面
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                              alpha=0.6, linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='f(x)')
        
        # 绘制搜索路径
        for i in range(self.pop_size):
            path = np.array([pos[i] for pos in self.history['positions']])
            ax.plot(path[:, 0], path[:, 1], [self.func(p) for p in path], 
                   '.-', markersize=5, linewidth=1, alpha=0.6)
        
        # 标记最优解
        best_path = np.array(self.history['x_best'])
        ax.plot(best_path[:, 0], best_path[:, 1], [self.func(p) for p in best_path], 
               'r*-', markersize=8, linewidth=2, label='Global Best')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x)')
        ax.set_title("Enhanced BAS Search Path on Ackley Function")
        ax.legend()
        plt.show()

# ------------------------------------------------------------------
# 3. 主程序示例
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 参数设置
    N_DIM = 2
    BOUNDS = (-5, 5)
    SEED = 42  # 固定随机种子
    
    # 初始化改进版BAS
    print("="*60)
    print("改进版天牛群算法 (Enhanced BAS) 优化")
    print("="*60)
    
    solver = EnhancedBAS(
        func=ackley_2d,
        n_dim=N_DIM,
        bounds=BOUNDS,
        pop_size=30,      # 种群大小 [5,6](@ref)
        step_max=2.0,     # 最大步长
        step_min=0.01,    # 最小步长
        d0=3.0,           # 初始触须间距
        max_iter=200,
        tol=1e-8,
        levy_scale=0.05,  # Lévy Flight缩放因子 [6](@ref)
        seed=SEED
    )
    
    # 运行优化
    x_opt, f_opt = solver.run()
    
    # 输出结果
    print("\n" + "="*60)
    print("优化结果总结")
    print("="*60)
    print(f"最优解 x* = {x_opt}")
    print(f"最优值 f(x*) = {f_opt:.10f}")
    print(f"总迭代次数 = {len(solver.history['f_best']) - 1}")
    
    # 可视化
    solver.plot_convergence()
    solver.plot_search_path_3d()