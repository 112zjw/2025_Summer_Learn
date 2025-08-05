import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Params:
    def __init__(self):
        self.r1 = 0.8
        self.r2 = 0.3
        self.lambda1 = 0.001
        self.lambda2 = 0.002
        self.K = 1000
        self.t_max = 200
        self.n_points = 1000
        self.eps = 1e-3  # 崩溃阈值

params = Params()

# ========== 单次模拟 ==========
def single_simulation(T, e1, e2, n_sim=40):
    catches = []
    for _ in range(n_sim):
        # 参数扰动 ±20%，但 e1,e2 已限制在 (0,1)
        r1 = params.r1 * np.random.uniform(0.8, 1.2)
        r2 = params.r2 * np.random.uniform(0.8, 1.2)
        λ1 = params.lambda1 * np.random.uniform(0.8, 1.2)
        λ2 = params.lambda2 * np.random.uniform(0.8, 1.2)

        def system(y, t):
            x1, x2 = y
            if x1 < params.eps or x2 < params.eps:
                return [0.0, 0.0]  # 提前终止
            z = 1.0 if x1 > T else 0.0
            dx1 = x1 * ((r1 - e1 * z) * (1 - x1 / params.K) - λ1 * x2)
            dx2 = x2 * (-(r2 + e2 * z) + λ2 * x1)
            return [dx1, dx2]

        t = np.linspace(0, params.t_max, params.n_points)
        sol = odeint(system, [800, 50], t)
        x1, x2 = sol[:, 0], sol[:, 1]

        # 若中途崩溃则记 0
        if np.any((x1 < params.eps) | (x2 < params.eps)):
            catches.append(0.0)
            continue

        dt = t[1] - t[0]
        z_vals = np.array([1.0 if xi > T else 0.0 for xi in x1])
        total = np.trapz(e1 * z_vals * x1 + e2 * z_vals * x2, dx=dt)
        catches.append(total)

    return np.mean(catches)

# ========== 主程序 ==========
def main():
    Ts  = np.linspace(400, 700, 11)          # 阈值 (单位个体)
    e1s = np.linspace(0.01, 0.99, 15)        # e1 ∈ (0,1)
    e2s = np.linspace(0.01, 0.99, 15)        # e2 ∈ (0,1)

    T_all, e1_all, e2_all, catch_all = [], [], [], []

    for T in tqdm(Ts, desc="阈值"):
        for e1 in e1s:
            for e2 in e2s:
                avg = single_simulation(T, e1, e2)
                T_all.append(T)
                e1_all.append(e1)
                e2_all.append(e2)
                catch_all.append(avg)

    df = pd.DataFrame({'T': T_all, 'e1': e1_all, 'e2': e2_all, '总捕捞量': catch_all})
    df.to_csv('Te1e2_无崩溃约束.csv', index=False, encoding='utf-8-sig')

    best = df.loc[df['总捕捞量'].idxmax()]
    print("\n最优组合（无崩溃，e1,e2∈(0,1)）")
    print("="*50)
    print(f"阈值 T = {best['T']:.1f}")
    print(f"e1   = {best['e1']:.2f}")
    print(f"e2   = {best['e2']:.2f}")
    print(f"总捕捞量 = {best['总捕捞量']:.2f}")

    # 三维可视化
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(T_all, e1_all, e2_all, c=catch_all, cmap='plasma', alpha=0.7)
    ax.scatter(best['T'], best['e1'], best['e2'],
               color='red', s=120, edgecolor='k', label='最优点')
    ax.set_xlabel('阈值 T')
    ax.set_ylabel('e1')
    ax.set_zlabel('e2')
    ax.set_title('(T, e1, e2) 与总捕捞量（无崩溃，e1,e2∈(0,1)）')
    fig.colorbar(sc, ax=ax, shrink=0.5, label='总捕捞量')
    ax.legend()
    plt.tight_layout()
    plt.savefig('3D_无崩溃.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()