import numpy as np
from scipy.optimize import linprog, minimize

# ========== 第一问：固定料场的运输优化 ==========
# 工地坐标与水泥需求
a = [1.25, 8.75, 0.5, 5.75, 3, 7.75]   # 工地横坐标
b = [1.25, 0.75, 4.75, 5, 6.5, 7.75]     # 工地纵坐标
d = [3, 5, 4, 7, 6, 11]                 # 水泥日需求量
d_total = sum(d)                        # 总需求量

# 临时料场坐标
x1_coord, y1_coord = 5, 1  # 料场A(5,1)
x2_coord, y2_coord = 2, 7  # 料场B(2,7)

# 计算距离
dist_to_x1 = [np.sqrt((x1_coord - a_i)**2 + (y1_coord - b_i)**2) for a_i, b_i in zip(a, b)]
dist_to_x2 = [np.sqrt((x2_coord - a_i)**2 + (y2_coord - b_i)**2) for a_i, b_i in zip(a, b)]

# 构造线性规划问题
f = dist_to_x1 + dist_to_x2  # 目标函数系数
A_ub = np.array([[1]*6 + [0]*6, [0]*6 + [1]*6])  # 料场供应能力约束
b_ub = [20, 20]                                   # 料场供应上限
A_eq = np.hstack([np.eye(6), np.eye(6)])          # 工地需求约束
b_eq = d                                         # 需求约束值

# 求解第一问
result1 = linprog(f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*12)
if not result1.success:
    raise ValueError("第一问求解失败: " + result1.message)

# 解析结果
transport_plan = result1.x
total_ton_km1 = result1.fun
print("========== 第一问结果 ==========")
print(f"总吨千米数: {total_ton_km1:.4f}")
print(f"料场A使用量: {sum(transport_plan[:6]):.2f}/20 吨")
print(f"料场B使用量: {sum(transport_plan[6:]):.2f}/20 吨")
print("\n详细供应方案:")
for i in range(6):
    from_a = transport_plan[i]
    from_b = transport_plan[i+6]
    print(f"工地{['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ'][i]}: {from_a:.2f}吨(A) + {from_b:.2f}吨(B) = {from_a+from_b:.2f}吨 (需{d[i]}吨)")

# ========== 第二问：单层优化方法 ==========
print("\n\n========== 第二问：单层优化方法 ==========")

# 将第一问的答案作为非线性规划的初始解
x0_lp = np.hstack([
    transport_plan,   # 第一问的运输方案 (12个变量)
    x1_coord, y1_coord,  # 料场A位置
    x2_coord, y2_coord   # 料场B位置
])

# 边界约束
# 运输量下界为0，无上界；位置约束在工地区域内
bounds_lp = [(0, None)] * 12 + [  
    (min(a), max(a)), (min(b), max(b)),  # 料场A位置边界
    (min(a), max(a)), (min(b), max(b))   # 料场B位置边界
]

# 定义目标函数 (吨千米数)
def objective(x):
    """目标函数：计算总吨千米数"""
    # 前12个元素是运输量：x[0]-x[5]为料场A到各工地的量
    #                x[6]-x[11]为料场B到各工地的量
    # 后4个元素是料场位置：x[12]和x[13]为料场A位置
    #                   x[14]和x[15]为料场B位置
    
    # 计算料场A到各工地的距离
    dist_A = [np.sqrt((x[12] - a_i)**2 + (x[13] - b_i)**2) for a_i, b_i in zip(a, b)]
    # 计算料场B到各工地的距离
    dist_B = [np.sqrt((x[14] - a_i)**2 + (x[15] - b_i)**2) for a_i, b_i in zip(a, b)]
    
    # 计算总吨千米数
    total_ton_km = sum(dist_A[i] * x[i] for i in range(6)) + sum(dist_B[i] * x[i+6] for i in range(6))
    return total_ton_km

# 定义不等式约束（料场供应能力）
def inequality_constraints(x):
    """不等式约束：两个料场的供应量不超过20吨"""
    depot_A_usage = sum(x[0:6])   # 料场A的总使用量
    depot_B_usage = sum(x[6:12])  # 料场B的总使用量
    return [20 - depot_A_usage, 20 - depot_B_usage]  # 供应量约束：使用量 ≤ 20

# 定义等式约束（工地需求）
def equality_constraints(x):
    """等式约束：各工地水泥需求满足"""
    constraints = []
    for i in range(6):
        # 该工地总接收量 = 料场A运量 + 料场B运量
        constraints.append(x[i] + x[i+6] - d[i])
    return constraints

# 约束条件设置
constraints = [
    {'type': 'ineq', 'fun': inequality_constraints},  # 不等式约束
    {'type': 'eq', 'fun': equality_constraints}       # 等式约束
]

# 求解非线性规划问题
res_lp = minimize(
    fun=objective, 
    x0=x0_lp,
    constraints=constraints,
    bounds=bounds_lp,
    method='SLSQP',  # 序列最小二乘规划法，适用于带约束问题
    options={'disp': True, 'maxiter': 1000}
)

if not res_lp.success:
    raise ValueError("第二问单层优化求解失败: " + res_lp.message)

# 解析结果
opt_solution = res_lp.x
opt_ton_km2 = res_lp.fun

# 提取运输方案
opt_transport_A = opt_solution[0:6]   # 料场A到各工地的运输量
opt_transport_B = opt_solution[6:12]  # 料场B到各工地的运输量
opt_depotA_pos = (opt_solution[12], opt_solution[13])  # 优化后料场A位置
opt_depotB_pos = (opt_solution[14], opt_solution[15])  # 优化后料场B位置

# 重新计算距离以验证结果
dist_opt_A = [np.sqrt((opt_depotA_pos[0] - a_i)**2 + (opt_depotA_pos[1] - b_i)**2) for a_i, b_i in zip(a, b)]
dist_opt_B = [np.sqrt((opt_depotB_pos[0] - a_i)**2 + (opt_depotB_pos[1] - b_i)**2) for a_i, b_i in zip(a, b)]

# 计算总吨千米数验证
verify_ton_km = sum(dist_opt_A[i] * opt_transport_A[i] for i in range(6)) + \
                sum(dist_opt_B[i] * opt_transport_B[i] for i in range(6))

# 输出结果
print("\n========== 第二问单层优化结果 ==========")
print(f"最优料场位置: 料场A({opt_depotA_pos[0]:.4f}, {opt_depotA_pos[1]:.4f})")
print(f"              料场B({opt_depotB_pos[0]:.4f}, {opt_depotB_pos[1]:.4f})")
print(f"最小总吨千米数: {opt_ton_km2:.4f} (验证: {verify_ton_km:.4f})")
print(f"节省吨千米数: {total_ton_km1 - opt_ton_km2:.4f} ({100*(total_ton_km1 - opt_ton_km2)/total_ton_km1:.2f}%)")
print(f"料场A使用量: {sum(opt_transport_A):.2f}/20 吨 ({(sum(opt_transport_A)/20)*100:.1f}%)")
print(f"料场B使用量: {sum(opt_transport_B):.2f}/20 吨 ({(sum(opt_transport_B)/20)*100:.1f}%)")

print("\n详细供应方案:")
for i in range(6):
    site = ['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ'][i]
    d1 = dist_opt_A[i]
    d2 = dist_opt_B[i]
    transport_c = opt_transport_A[i]
    transport_d = opt_transport_B[i]
    site_ton_km = transport_c * d1 + transport_d * d2
    
    print(f"工地{site}:")
    print(f"  料场A: {transport_c:.2f}吨 × {d1:.3f}km = {transport_c*d1:.3f}吨公里")
    print(f"  料场B: {transport_d:.2f}吨 × {d2:.3f}km = {transport_d*d2:.3f}吨公里")
    print(f"  总供应: {transport_c+transport_d:.2f}吨 (需{d[i]}吨), 运输成本: {site_ton_km:.3f}吨公里")
    print("-"*50)

# ===== 错误检查 =====
print("\n===== 错误检查 =====")

# 1. 总需求满足检查
total_supply = sum(opt_transport_A) + sum(opt_transport_B)
if abs(total_supply - d_total) > 1e-5:
    print(f"错误: 总供应量({total_supply:.2f}吨) ≠ 总需求量({d_total}吨)")
else:
    print("√ 总供应量满足总需求")

# 2. 各工地需求检查
for i in range(6):
    supplied = opt_transport_A[i] + opt_transport_B[i]
    if abs(supplied - d[i]) > 1e-5:
        print(f"错误: 工地{['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ'][i]}供应量({supplied:.2f}吨) ≠ 需求量({d[i]}吨)")
    else:
        print(f"√ 工地{['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ'][i]}供应满足")

# 3. 料场能力检查
depotA_usage = sum(opt_transport_A)
depotB_usage = sum(opt_transport_B)
if depotA_usage > 20 + 1e-5:
    print(f"错误: 料场A供应量({depotA_usage:.2f}吨) > 20吨上限")
else:
    print(f"√ 料场A供应量在限制内")
    
if depotB_usage > 20 + 1e-5:
    print(f"错误: 料场B供应量({depotB_usage:.2f}吨) > 20吨上限")
else:
    print(f"√ 料场B供应量在限制内")

# 4. 位置边界检查
x_min, x_max = min(a), max(a)
y_min, y_max = min(b), max(b)

if not (x_min <= opt_depotA_pos[0] <= x_max) or not (y_min <= opt_depotA_pos[1] <= y_max):
    print(f"警告: 料场A位置({opt_depotA_pos[0]:.4f}, {opt_depotA_pos[1]:.4f})超出工地区域")
    
if not (x_min <= opt_depotB_pos[0] <= x_max) or not (y_min <= opt_depotB_pos[1] <= y_max):
    print(f"警告: 料场B位置({opt_depotB_pos[0]:.4f}, {opt_depotB_pos[1]:.4f})超出工地区域")

# 5. 结果一致性检查
if abs(opt_ton_km2 - verify_ton_km) > 1e-5:
    print(f"错误: 优化目标值({opt_ton_km2:.4f}) ≠ 计算结果({verify_ton_km:.4f})")
else:
    print("√ 优化目标值一致")

# 无错误则显示通过
if all([
    abs(total_supply - d_total) <= 1e-5,
    depotA_usage <= 20 + 1e-5,
    depotB_usage <= 20 + 1e-5,
    abs(opt_ton_km2 - verify_ton_km) <= 1e-5
]):
    print("\n√√√ 所有检查通过，无错误 √√√")