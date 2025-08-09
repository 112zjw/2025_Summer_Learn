import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from scipy.optimize import minimize

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
# 量子混沌初始化种群 - 优化种群初始化方式
# =========================================
def quantum_chaos_init(pop_size, dim, lb, ub):
    """
    使用量子混沌映射初始化种群，提高种群多样性
    改进方向1:优化种群初始化方式 - 量子计算
    """
    # 初始化种群
    population = np.zeros((pop_size, dim))
    
    # 量子混沌映射参数
    beta = 0.5
    u = 3.9  # 混沌参数
    
    for i in range(pop_size):
        # 混沌序列初始化
        chaos_seq = np.zeros(dim)
        chaos_seq[0] = np.random.rand()
        
        # 生成混沌序列
        for j in range(1, dim):
            chaos_seq[j] = u * chaos_seq[j-1] * (1 - chaos_seq[j-1])
        
        # 量子旋转门
        quantum_state = np.exp(1j * 2 * np.pi * chaos_seq)
        
        # 转换为实数解
        population[i] = lb + (ub - lb) * np.abs(quantum_state.real)
    
    return population

# =========================================
# 改进的鲸鱼优化算法
# =========================================
def improved_whale_optimization_algorithm(obj_func, dim=30, lb=-100, ub=100, 
                                          pop_size=30, max_iter=500, b=1, 
                                          mutation_prob=0.05, sim_prob=0.3):
    """
    改进的鲸鱼优化算法，包含五个优化方向：
    
    1.优化种群初始化方式: 使用量子混沌初始化替代随机初始化
    2.优化模式转换策略: 非线性变化的p阈值
    3.丰富全局勘探策略: 结合PSO算法优化包围猎物步骤
    4.丰富局部开发策略: 引入单纯形搜索法优化局部开发
    5.增加跳出局部最优能力: 加入平均差分变异策略
    
    参数:
    - obj_func: 目标函数
    - dim: 优化参数维度
    - lb, ub: 参数取值范围
    - pop_size: 种群数量
    - max_iter: 最大迭代次数
    - b: 螺旋形状常数
    - mutation_prob: 变异概率
    - sim_prob: 单纯形法应用概率
    
    返回:
    - best_solution: 最优解
    - convergence_curve: 收敛曲线
    - diversity_history: 种群多样性历史记录
    - best_position_history: 每次迭代的最优位置
    """
    # 1. 初始化种群 - 使用量子混沌初始化
    # 改进方向1:优化种群初始化方式 (量子计算)
    positions = quantum_chaos_init(pop_size, dim, lb, ub)
    
    # 记录个体历史最优位置和适应度
    pbest_positions = positions.copy()
    pbest_fitness = np.array([obj_func(ind) for ind in positions])
    
    # 2. 计算初始适应度并记录最优个体
    fitness = np.array([obj_func(ind) for ind in positions])
    best_index = np.argmin(fitness)
    best_solution = positions[best_index].copy()
    best_fitness = fitness[best_index]
    
    # 存储收敛曲线
    convergence_curve = np.zeros(max_iter)
    
    # 存储种群多样性历史
    diversity_history = np.zeros(max_iter)
    
    # 存储每次迭代的最优位置
    best_position_history = np.zeros((max_iter, dim))
    
    # 3. 主循环
    for iter_num in range(max_iter):
        # 4. 优化模式转换策略 (非线性变化)
        # 改进方向2:优化模式转换策略 (非线性变化)
        a = 2 * (1 - iter_num / max_iter)**1.5  # 非线性衰减系数
        p_threshold = 0.5 * (1 - iter_num / max_iter)  # 非线性变化的p阈值
        
        # 记录种群多样性
        diversity_history[iter_num] = np.mean(np.std(positions, axis=0))
        
        # 存储本次迭代的最优位置
        best_position_history[iter_num] = best_solution.copy()
        
        # 主循环内个体更新
        for i in range(pop_size):
            # 更新参数
            r = np.random.rand(dim)  # [0,1]随机向量
            A = 2 * a * r - a        # 计算A向量
            C = 2 * r                # 计算C向量
            p = np.random.rand()      # [0,1]随机数
            l = np.random.uniform(-1, 1, dim)  # [-1,1]随机数
            
            # 行为选择 - 非线性模式转换
            if p < p_threshold:
                # 包围或随机搜索
                if np.all(np.abs(A) < 1):
                    # 包围猎物（引入PSO思想）
                    # 改进方向3:丰富全局勘探策略 (结合PSO算法)
                    c1 = 1.5  # 局部学习因子
                    c2 = 1.5  # 全局学习因子
                    D_pbest = np.abs(pbest_positions[i] - positions[i])
                    D_gbest = np.abs(best_solution - positions[i])
                    
                    # 结合PSO的更新策略
                    positions[i] = positions[i] + c1 * np.random.rand() * D_pbest + c2 * np.random.rand() * D_gbest
                else:
                    # 随机搜索（全局勘探）
                    rand_index = np.random.randint(pop_size)
                    D = np.abs(C * positions[rand_index] - positions[i])
                    positions[i] = positions[rand_index] - A * D
            else:
                # 气泡网攻击（局部开发）
                D = np.abs(best_solution - positions[i])
                
                # 使用单纯形法改进局部搜索 (一定概率)
                # 改进方向4:丰富局部开发策略 (结合单纯形法)
                if np.random.rand() < sim_prob:
                    # 选择三个不同点构建单纯形
                    idx = np.random.choice(pop_size, 3, replace=False)
                    simplex = positions[idx]
                    
                    # 计算质心（排除最差点）
                    worst_idx = np.argmax([obj_func(p) for p in simplex])
                    centroid = np.mean(np.delete(simplex, worst_idx, axis=0), axis=0)
                    
                    # 反射操作
                    alpha = 1.0
                    reflection = centroid + alpha * (centroid - positions[i])
                    
                    # 边界处理
                    reflection = np.clip(reflection, lb, ub)
                    
                    # 如果反射点更好，则替换当前位置
                    if obj_func(reflection) < obj_func(positions[i]):
                        positions[i] = reflection
                
                # 原始螺旋更新
                positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
            
            # 边界处理
            positions[i] = np.clip(positions[i], lb, ub)
            
            # 计算新适应度
            new_fitness = obj_func(positions[i])
            
            # 更新个体最优
            if new_fitness < pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = new_fitness
            
            # 更新全局最优
            if new_fitness < best_fitness:
                best_solution = positions[i].copy()
                best_fitness = new_fitness
        
        # 5. 增加跳出局部最优能力（平均差分变异）
        # 改进方向5:增加跳出局部最优能力 (平均差分变异)
        if np.random.rand() < mutation_prob:
            # 计算平均差向量
            center = np.mean(positions, axis=0)
            diff_vector = np.mean([p - center for p in positions], axis=0)
            
            # 差分变异操作
            mutation_indices = np.random.choice(pop_size, int(pop_size * 0.3), replace=False)
            for idx in mutation_indices:
                positions[idx] = positions[idx] + 0.5 * diff_vector
            
                # 边界处理
                positions[idx] = np.clip(positions[idx], lb, ub)
                
                # 更新适应度
                fitness[idx] = obj_func(positions[idx])
                
                # 更新个体最优
                if fitness[idx] < pbest_fitness[idx]:
                    pbest_positions[idx] = positions[idx].copy()
                    pbest_fitness[idx] = fitness[idx]
                
                # 更新全局最优
                if fitness[idx] < best_fitness:
                    best_solution = positions[idx].copy()
                    best_fitness = fitness[idx]
        
        # 记录本次迭代的最优值
        convergence_curve[iter_num] = best_fitness
    
    return best_solution, convergence_curve, diversity_history, best_position_history

# =========================================
# 算法执行与可视化
# =========================================
def visualize_results(best_solution, convergence_curve, diversity_history, best_position_history):
    """可视化结果"""
    max_iter = len(convergence_curve)
    dim = len(best_solution)
    
    # 创建可视化图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('改进鲸鱼优化算法(WOA)结果可视化', fontsize=18, y=0.98)
    
    # 1. 收敛曲线
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(convergence_curve, 'b-', linewidth=2, label='改进WOA')
    ax1.set_title('算法收敛曲线', fontsize=15)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('适应度(F1值)', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12)
    
    # 2. 种群多样性分析
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(diversity_history, 'g-', linewidth=2)
    ax2.set_title('种群多样性变化', fontsize=15)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('多样性指标', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # 3. 最优解维度分布
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(range(1, dim+1), best_solution, color='skyblue')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_title('最优解的维度分布', fontsize=15)
    ax3.set_xlabel('参数维度', fontsize=12)
    ax3.set_ylabel('参数值', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 最优位置在二维子空间的演化路径（选择前两个维度）
    ax4 = plt.subplot(2, 2, 4)
    
    # 创建热力图色彩
    colors = cm.viridis(np.linspace(0, 1, max_iter))
    
    # 绘制最优位置路径
    for i in range(1, max_iter):
        ax4.plot([best_position_history[i-1][0], best_position_history[i][0]], 
                 [best_position_history[i-1][1], best_position_history[i][1]], 
                 color=colors[i], linewidth=1.5, alpha=0.7)
    
    # 添加起点和终点标记
    ax4.scatter(best_position_history[0][0], best_position_history[0][1], 
                s=100, c='green', marker='o', edgecolor='k', zorder=10, label='起始位置')
    ax4.scatter(best_position_history[-1][0], best_position_history[-1][1], 
                s=100, c='red', marker='*', edgecolor='k', zorder=10, label='最优位置')
    
    # 绘制优化目标函数的等高线（仅显示前两个维度）
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f1(np.array([X[i, j], Y[i, j], *np.zeros(dim-2)]))
    
    # 绘制等高线
    contour = ax4.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.6)
    fig.colorbar(contour, ax=ax4, shrink=0.8)
    ax4.set_title('最优位置在二维子空间的演化路径', fontsize=15)
    ax4.set_xlabel('维度1', fontsize=12)
    ax4.set_ylabel('维度2', fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('Improved_WOA_Results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出详细分析结果
    print(f"\n{'改进鲸鱼优化算法分析报告':^60}")
    print("="*60)
    print(f"优化函数: F1 = ∑x_i² (维度={dim})")
    print(f"搜索范围: [{lb}, {ub}]^n")
    print(f"种群大小: {pop_size}, 迭代次数: {max_iter}")
    print("\n优化结果:")
    print(f"找到最优值: {best_fitness:.10e}")
    print(f"实际最小值应该接近于: 0.0")
    
    # 显示前10个维度的最优值
    print("\n最优解前10个维度:")
    for i in range(min(dim, 10)):
        print(f"维度{i+1}: {best_solution[i]:.6e}", end="\t")
        if (i+1) % 5 == 0:
            print()
    
    # 计算误差百分比
    actual_min = 0
    error_percentage = (best_fitness - actual_min) / (actual_min + 1e-16) * 100
    print(f"\n\n误差百分比: {error_percentage:.8f}%")
    
    # 结果质量评估
    solution_norm = np.linalg.norm(best_solution)
    solution_std = np.std(best_solution)
    
    print("\n结果质量评估:")
    print(f"最优位置范数: {solution_norm:.4f}")
    print(f"最优位置标准差: {solution_std:.4e}")
    print(f"初始多样性: {diversity_history[0]:.4f}")
    print(f"最终多样性: {diversity_history[-1]:.4f}")
    
    return fig

# =========================================
# 主程序
# =========================================
if __name__ == "__main__":
    # 参数设置
    dim = 30
    lb, ub = -100, 100
    pop_size = 50
    max_iter = 
    
    # 执行改进的鲸鱼优化算法
    best_solution, convergence_curve, diversity_history, best_position_history = \
        improved_whale_optimization_algorithm(
            obj_func=f1, dim=dim, lb=lb, ub=ub, 
            pop_size=pop_size, max_iter=max_iter
        )
    
    best_fitness = np.min(convergence_curve)
    
    # 结果可视化
    visualize_results(
        best_solution, convergence_curve, 
        diversity_history, best_position_history
    )