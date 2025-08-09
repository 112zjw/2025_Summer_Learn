"""
模拟退火算法（Simulated Annealing, SA）演示
问题：在区间 [-10, 10] 内找 f(x)=x^2 的最小值
"""

import random
import math

# ========= 1. 基本参数 =========
T0          = 100.0   # 初始温度
T_MIN       = 1e-3    # 最低温度（停止条件）
ALPHA       = 0.95    # 温度衰减系数：T_k+1 = ALPHA * T_k
MARKOV_LEN  = 100     # 每个温度下的内循环次数（Markov 链长度）
BOUND       = (-10, 10)  # 搜索区间

# ========= 2. 目标函数 =========
def objective(x):
    """目标函数：f(x)=x^2，需要最小化"""
    return x ** 2

# ========= 3. 产生新解 =========
def neighbor(x, bound, temperature):
    """
    在当前解 x 的邻域内随机产生新解。
    这里使用高斯扰动，扰动幅度随温度降低而减小。
    """
    low, high = bound
    # 高斯扰动：均值=当前解，标准差=temperature（温度高时扰动大）
    x_new = random.gauss(x, temperature)
    # 保证仍在合法区间内
    x_new = max(low, min(high, x_new))
    return x_new

# ========= 4. 接受准则 =========
def accept(delta, temperature):
    """
    Metropolis 准则：
    如果 delta < 0（新解更好），则一定接受；
    如果 delta >= 0，以概率 exp(-delta / temperature) 接受。
    """
    if delta < 0:
        return True
    else:
        p = math.exp(-delta / temperature)
        return random.random() < p

# ========= 5. 主循环：模拟退火 =========
def simulated_annealing():
    # 随机产生初始解
    x_cur = random.uniform(*BOUND)
    f_cur = objective(x_cur)
    best_x, best_f = x_cur, f_cur

    # 记录迭代过程（可选）
    history = [(x_cur, f_cur)]

    T = T0
    step = 0
    while T > T_MIN:
        for _ in range(MARKOV_LEN):
            # 产生邻域内新解
            x_new = neighbor(x_cur, BOUND, T)
            f_new = objective(x_new)
            delta = f_new - f_cur

            # 根据 Metropolis 准则决定是否接受
            if accept(delta, T):
                x_cur, f_cur = x_new, f_new
                # 更新全局最优
                if f_cur < best_f:
                    best_x, best_f = x_cur, f_cur

            history.append((x_cur, f_cur))
            step += 1

        # 降温
        T *= ALPHA

        # 打印当前温度下的信息
        print(f"Step={step:>4d} | T={T:>8.4f} | cur_x={x_cur:>8.4f} | cur_f={f_cur:>8.4f}")

    return best_x, best_f, history

# ========= 6. 运行并输出结果 =========
if __name__ == "__main__":
    best_x, best_f, history = simulated_annealing()
    print("\n==== 最终结果 ====")
    print(f"最优解 x = {best_x:.6f}")
    print(f"最小值 f = {best_f:.6f}")