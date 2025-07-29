import numpy as np
from scipy.optimize import minimize
#目标函数
def fun(x):
    a=np.array([1,4,3,5,9,12,6,20,17,8])
    b=np.array([2,10,8,18,1,4,5,10,8,9])
    f=np.zeros(10)
    for i in range(10):
        f[i]=np.abs(x[0]-a[i]) + np.abs(x[1]-b[i])
    return f   


#总的目标函数，取目标函数值数组中的最大值
def overall_objective(x):
    return np.max(fun(x))


#初始值，对应MatLab中的x0
x0=np.array([6,6])
#决策变量的下界，对应MatLab中的Lb
lb=np.array([3,4])
#决策变量的上界，对应MatLab中的ub
ub=np.array([8,10])

#约束条件，这里定义变量的边界约束
bounds=[(lb[0],ub[0]),(lb[1],ub[1])]

#使用minimize进行优化，采用SLSQP方法，和MatLab示例中功能类似进行最小最大优化
result =minimize(overall_objective,x0,method='SLSQP',bounds=bounds)

x= result.x
feval = fun(x)
print("优化后的坐标点x:")
print(x)
print("对应的各个目标函数值feval：")
print(feval)
print("最小的最大距离（取feval中的最大值）:")
print(np.max(feval))