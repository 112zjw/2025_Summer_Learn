import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FireflyAlgorithm:
    """
    萤火虫优化算法实现 (Firefly Algorithm)
    参考文献：
    [1] Yang, X. S. (2008). Nature-inspired metaheuristic algorithms. Luniver press. [1,6](@ref)
    [2] Yang, X. S. (2009). Firefly algorithms for multimodal optimization. In International symposium on stochastic algorithms (pp. 169-178). [1](@ref)
    """
    
    def __init__(self, func, dim, lb, ub, 
                 pop_size=40, max_iter=200, 
                 alpha=0.5, beta0=1.0, gamma=0.1):
        """
        初始化算法参数
        :param func: 目标函数
        :param dim: 问题维度
        :param lb: 变量下界 (标量或数组)
        :param ub: 变量上界 (标量或数组)
        :param pop_size: 种群大小 [4,6](@ref)
        :param max_iter: 最大迭代次数
        :param alpha: 随机扰动系数 (初始值) [1](@ref)
        :param beta0: 最大吸引度 [1,4](@ref)
        :param gamma: 光吸收系数 [6](@ref)
        """
        self.func = func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else np.full(dim, lb)
        self.ub = np.array(ub) if isinstance(ub, list) else np.full(dim, ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        # 初始化种群
        self.fireflies = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.brightness = np.apply_along_axis(func, 1, self.fireflies)
        
        # 跟踪最优解
        self.best_idx = np.argmin(self.brightness)
        self.best_firefly = self.fireflies[self.best_idx].copy()
        self.best_brightness = self.brightness[self.best_idx]
        
        # 收敛曲线记录
        self.convergence_curve = np.zeros(max_iter)
    
    def _distance(self, a, b):
        """计算欧氏距离 [6](@ref)"""
        return np.linalg.norm(a - b)
    
    def _attractiveness(self, r):
        """计算吸引力 (随距离衰减) [1,4](@ref)"""
        return self.beta0 * np.exp(-self.gamma * r**2)
    
    def optimize(self):
        """执行优化主循环 [4](@ref)"""
        for iter in range(self.max_iter):
            # ===== 新增：图片中的参数自适应策略 =====
            # 1. α 的线性衰减 (匹配图片公式)
            current_alpha = self.alpha * (1 - iter/self.max_iter)
            
            # 2. β₀ 的随机重置 (匹配图片公式)
            if np.random.rand() < 0.5:  # 50%概率重置
                current_beta0 = 0.5 + np.random.rand()  # [0.5, 1.5]范围
            else:                       # 50%概率保持
                current_beta0 = self.beta0
            
           # ===== 原核心优化循环 =====
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if self.brightness[j] < self.brightness[i]:
                        r = self._distance(self.fireflies[i], self.fireflies[j])
                        
                        # 使用自适应参数（关键修改）
                        beta = current_beta0 * np.exp(-self.gamma * r**2)  # 修改了β₀
                        
                        rand_vec = current_alpha * (np.random.rand(self.dim) - 0.5)  # 修改了α
                        self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + rand_vec
                        
                        # 边界约束处理
                        self.fireflies[i] = np.clip(self.fireflies[i], self.lb, self.ub)
                        
                        # 更新亮度
                        new_brightness = self.func(self.fireflies[i])
                        self.brightness[i] = new_brightness
                        
                        # 更新全局最优
                        if new_brightness < self.best_brightness:
                            self.best_brightness = new_brightness
                            self.best_firefly = self.fireflies[i].copy()
            
            # 记录当前最优值
            self.convergence_curve[iter] = self.best_brightness
            print(f"Iter {iter+1}/{self.max_iter}: Best = {self.best_brightness:.6f}")
        
        return self.best_firefly, self.best_brightness
    
    def plot_convergence(self):
        """绘制收敛曲线 [4](@ref)"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, self.max_iter+1), self.convergence_curve, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Best Function Value (log scale)')
        plt.title('Firefly Algorithm Convergence')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    def plot_search_path(self, func_3d=None):
        """
        可视化2D/3D搜索路径 (仅支持2-3维问题)
        :param func_3d: 3D目标函数 (仅当dim=2时生效)
        """
        if self.dim not in [2, 3]:
            print("Warning: Visualization only for 2D/3D problems")
            return
            
        fig = plt.figure(figsize=(12, 9))
        
        if self.dim == 2:
            # 创建等高线图背景
            x = np.linspace(self.lb[0], self.ub[0], 100)
            y = np.linspace(self.lb[1], self.ub[1], 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.func([X[i, j], Y[i, j]])
            
            plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.6)
            plt.colorbar(label='Function Value')
            
            # 绘制萤火虫路径
            for i in range(self.pop_size):
                plt.scatter(self.fireflies[i, 0], self.fireflies[i, 1], 
                           c='r', s=30, alpha=0.7)
            
            plt.scatter(self.best_firefly[0], self.best_firefly[1], 
                       s=150, c='gold', marker='*', edgecolor='k', label='Best Solution')
            plt.xlabel('x1')
            plt.ylabel('x2')
            
        elif self.dim == 3 and func_3d:
            # 3D可视化
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制最优解路径
            ax.scatter(self.best_firefly[0], self.best_firefly[1], self.best_firefly[2], 
                      s=150, c='gold', marker='*', edgecolor='k', label='Best Solution')
            
            # 绘制种群分布
            ax.scatter(self.fireflies[:, 0], self.fireflies[:, 1], self.fireflies[:, 2],
                      c='r', s=30, alpha=0.7, label='Fireflies')
            
            # 创建函数曲面
            x = np.linspace(self.lb[0], self.ub[0], 30)
            y = np.linspace(self.lb[1], self.ub[1], 30)
            X, Y = np.meshgrid(x, y)
            Z = func_3d(X, Y)
            
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('f(x)')
        
        plt.title('Firefly Search Space')
        plt.legend()
        plt.show()


# ====================== 测试与示例 ======================
if __name__ == "__main__":
    # 1. 定义测试函数 (Sphere函数)
    def sphere(x):
        return np.sum(np.array(x)**2)
    
    # 2. 定义3D可视化函数 (仅用于3D展示)
    def sphere_3d(x, y):
        return x**2 + y**2
    
    # 3. 算法参数设置
    dim = 2
    lb, ub = -5, 5
    max_iter = 100
    
    # 4. 运行优化
    print("="*50)
    print("萤火虫算法优化示例")
    print("="*50)
    fa = FireflyAlgorithm(
        func=sphere,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=30,
        max_iter=max_iter,
        alpha=0.3,
        beta0=1.0,
        gamma=0.1
    )
    
    best_solution, best_value = fa.optimize()
    
    # 5. 输出结果
    print("\n" + "="*50)
    print("优化结果")
    print("="*50)
    print(f"最优解: {best_solution}")
    print(f"最优值: {best_value:.10f}")
    
    # 6. 可视化
    fa.plot_convergence()
    
    # 根据维度选择可视化方式
    if dim == 2:
        fa.plot_search_path()
    elif dim == 3:
        fa.plot_search_path(sphere_3d)