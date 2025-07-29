import sympy as sp  # 导入符号计算库

# 定义符号变量：x为未知量，α,β,θ,D₀为方程参数
x, alpha, beta, D0, theta = sp.symbols('x alpha beta D0 theta')

# 构建方程：
# 左边：x² * (1 + 1/tan²β)
# 右边：(x·tanα + D₀)² * tan²(θ/2)
equation = sp.Eq(
    x**2 * (1 + 1/sp.tan(beta)**2), 
    (x * sp.tan(alpha) + D0)**2 * sp.tan(theta/2)**2
)

# 对方程求解x（符号解析解）
solutions = sp.solve(equation, x)

# 打印解（通常有多个根）
print("解析解为：", solutions)