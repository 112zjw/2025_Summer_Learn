import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 目标函数定义（基于图片1的F1函数）
# =========================================
def f1(x):
    """F1函数：n维平方和函数（n=30）"""
    return np.sum(x**2)

# =========================================
# 鲸鱼优化算法实现（基于图片2和3的流程图和原理）
# =========================================
def whale_optimization_algorithm(obj_func, dim=30, lb=-100, ub=100, 
                                 pop_size=50, max_iter=500, b=1):
    """
    参数说明（基于图片1的算法参数）:
    - obj_func: 目标函数
    - dim: 优化参数维度 (n=30)
    - lb, ub: 参数取值范围 (-100, 100)
    - pop_size: 种群数量 (2)
    - max_iter: 最大迭代次数 (20)
    - b: 螺旋形状常数（图片3中b=1）
    
    返回:
    - best_solution: 最优解
    - convergence_curve: 收敛曲线（每次迭代的最优适应度）
    """
    # 1. 初始化种群（图片2：随机产生初始个体）
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    
    # 2. 计算初始适应度并记录最优个体（图片2）
    fitness = np.array([obj_func(ind) for ind in positions])
    best_index = np.argmin(fitness)
    best_solution = positions[best_index].copy()
    best_fitness = fitness[best_index]
    
    # 存储收敛曲线
    convergence_curve = np.zeros(max_iter)
    
    # 3. 主循环（图片2流程图）
    for iter_num in range(max_iter):
        # 线性减小a值（图片3：从2线性减小到0）
        a = 2 * (1 - iter_num / max_iter)
        
        for i in range(pop_size):
            # 4. 更新参数（图片3算法变量）
            r = np.random.rand(dim)  # [0,1]随机向量
            A = 2 * a * r - a        # 计算A向量
            C = 2 * r                # 计算C向量
            p = np.random.rand()      # [0,1]随机数
            l = np.random.uniform(-1, 1, dim)  # [-1,1]随机数
            
            # 5. 行为选择（图片2决策流程）
            if p < 0.5:
                # 包围或随机搜索（图片3的全局勘探）
                if np.all(np.abs(A) < 1):
                    # 包围猎物（靠近当前最优）
                    D = np.abs(C * best_solution - positions[i])
                    positions[i] = best_solution - A * D
                else:
                    # 随机搜索（全局勘探）
                    rand_index = np.random.randint(pop_size)
                    D = np.abs(C * positions[rand_index] - positions[i])
                    positions[i] = positions[rand_index] - A * D
            else:
                # 气泡网攻击（图片3的局部开发）
                D = np.abs(best_solution - positions[i])
                positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
            
            # 6. 边界处理（确保在[-100,100]范围内）
            positions[i] = np.clip(positions[i], lb, ub)
            
            # 7. 更新适应度（图片2：计算个体适应度）
            new_fitness = obj_func(positions[i])
            
            # 8. 更新最优解（图片2：记录最优个体）
            if new_fitness < best_fitness:
                best_solution = positions[i].copy()
                best_fitness = new_fitness
        
        # 记录本次迭代的最优值
        convergence_curve[iter_num] = best_fitness
    
    return best_solution, convergence_curve

# =========================================
# 算法执行与可视化
# =========================================
if __name__ == "__main__":
    # 执行鲸鱼优化算法
    best_solution, convergence_curve = whale_optimization_algorithm(
        obj_func=f1, dim=30, lb=-100, ub=100, pop_size=2, max_iter=20
    )
    
    # 结果输出
    print(f"最优解: {best_solution}")
    print(f"最小值: {np.min(convergence_curve):.6f}")
    print(f"最优位置: [{', '.join(f'{x:.3f}' for x in best_solution)}]")
    
    # 创建可视化图表
    plt.figure(figsize=(12, 10))
    
    # 1. 收敛曲线（类似图片3右上的折线图）
    plt.subplot(2, 1, 1)
    plt.plot(convergence_curve, 'o-', linewidth=2)
    plt.title('WOA算法收敛曲线', fontsize=14)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度(F1值)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.yscale('log')  # 使用对数刻度显示更大范围
    
    # 2. 最优解维度分布（显示30维参数）
    plt.subplot(2, 1, 2)
    plt.bar(range(1, 31), best_solution, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title('最优解的维度分布', fontsize=14)
    plt.xlabel('参数维度', fontsize=12)
    plt.ylabel('参数值', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('WOA_optimization_results.png', dpi=300)
    plt.show()