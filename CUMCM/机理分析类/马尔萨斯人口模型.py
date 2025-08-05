import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. 定义马尔萨斯人口模型的微分方程
def malthusian_model(P, t, r):
    dPdt = r * P
    return dPdt
# 在定义微分方程函数时，时间变量 t 必须放在第二个位置；从第三个位置开始才是你要额外传递的参数。


# 2. 参数设置
r = 0.001          # 人口增长率
P0 = 100           # 初始人口数量
t = np.linspace(0, 1000, 1000)  # 时间范围

# 3. 求解微分方程
P = odeint(malthusian_model, P0, t, args=(r,))

# 4. 绘图
plt.plot(t, P)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Malthusian Population Model')
plt.grid(True)
plt.show()

seaborn美化




