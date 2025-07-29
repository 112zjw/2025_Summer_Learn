from pulp import LpMaximize, LpProblem, LpVariable,lpSum, value
#创建最大化问题
problem = LpProblem( "Knapsack_Problem",LpMaximize)
#定义目标函数系数（利润）
profits =[540,200,180,350,60,150,280,450,320,120]
#定义约束系数（重量）
weights=[6,3,4,5,1,2,3,5,4,2]
max_weight = 30
#定义决策变量（0-1变量）
x = [LpVariable( f"x{i+1}", cat="Binary") for i in range(10)]

#定义目标函数（最大化总利润）
problem += lpSum(profits[i] *x[i] for i in range(10)),"Total_Profit"

#添加约束条件（总重量不超过最大承载量）
problem +=lpSum(weights[i] *x[i] for i in range(10)) <= max_weight,"Weight_Constraint"
#求解问题
problem.solve()
#输出结果
print("选中的物品分别为（0表示未选，1表示选中）：")
selected_items =[int(value(x[i])) for i in range(10)]
print(selected_items)
print("最终最大利润为：")
print(value(problem.objective))