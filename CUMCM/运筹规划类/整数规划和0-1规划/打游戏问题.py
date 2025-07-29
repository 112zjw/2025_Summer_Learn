from pulp import LpMaximize,LpProblem,LpVariable,lpSum,value
#创建一个最大化问题
problem = LpProblem( "Maximize_Experience", LpMaximize)
#定义决策变量
x1=LpVariable("x1",lowBound=0,cat="Integer")#A 图通关次数
x2=LpVariable("x2",lowBound=0,cat="Integer")#B 图通关次数
x3=LpVariable("x3",lowBound=0,cat="Integer")#C 图通关次数
#定义目标函数
problem +=20 *x1 +30 *x2 +40 *x3,"Total_Experience"
#添加约束条件
problem +=4*x1+8*x2+10*x3<= 100,"Resource_Constraint"
problem += x1 + x2 + x3 <= 20,"Time_Constraint"
#求解问题
problem.solve()
#输出结果
print("A、B、C三图分别通关的次数为：")
print(int(value(x1)),int(value(x2)),int(value(x3)))
print("最终获得的经验为：")
print(int(value(problem.objective)))