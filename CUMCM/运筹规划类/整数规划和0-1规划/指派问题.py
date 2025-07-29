from pulp import LpMinimize,LpProblem,LpVariable,lpSum,value
#创建最小化问题
problem = LpProblem("Assignment_Problem",LpMinimize)
#定义目标函数系数（泳姿时间）
c = [66.8, 75.6,87, 58.6, 57.2, 66,66.4,53,78,67.8,84.6, 59.4,
    70,74.2, 69.6, 57.2, 67.4, 71,83.8, 62.4]
#定义决策变量（0-1变量）
x=[LpVariable(f"x{i+1}",cat="Binary") for i in range(20)]
#定义目标函数（最小化总时间）
problem +=lpSum(c[i] *x[i] for i in range(20)),"Total_Time"
#添加不等式约束（每个人只能入选四种泳姿之一）
A=[
[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
]

b=[1,1,1,1,1]
for i in range(len(A)):
    problem += lpSum(A[i][j] *x[j] for j in range(20)) <= b[i], f"Person_Constraint_{i+1}"
#添加等式约束（每种泳姿有且仅有一人参加）
Aeq=[
[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
[0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
]
beq =[1, 1, 1, 1]
for i in range(len(Aeq)):
    problem += lpSum(Aeq[i][j] * x[j] for j in range(20)) == beq[i], f"Style_Constraint_{i+1}"
#求解问题
problem.solve()

print("分配结果（0表示未选，1表示选中）：")
assignments =[int(value(x[i])) for i in range(20)]
print(assignments)
print("最终最小总时间为：")
print(value(problem.objective))
#可选：将结果重塑为4x5矩阵形式，方便查看
import numpy as np
assignment_matrix = np.array(assignments).reshape(5,4)
print("分配矩阵（行对应泳姿，列对应队员）：")
print(assignment_matrix)