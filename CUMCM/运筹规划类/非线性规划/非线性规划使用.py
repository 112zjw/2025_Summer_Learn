import numpy as np
from scipy.optimize import minimize

#目标函数，接受一个长度为2的一维数组x（代表两个变量x1和x2），计算并返回目标函数值

def fun1(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-2*x[0]-5*x[1]
#第一个非线性约束条件函数，接受变量x，返回对应约束条件的值

def nonlcon1(x):
    return -(x[0]-1)**2+x[1]
#第二个非线性约束条件函数，接受变量x，返回对应约束条件的值

def nonlcon2(x):
    return 2*x[0]-3*x[1]+6
#例题1的求解
#初始值设定，设置为一个二维的numpy数组，初始值为[0，θ]
x0=np.array([0,0])
#使用默认算法（内点法类似的算法，在scipy中会根据情况选择合适的）求解

con = ({'type':'ineq','fun':nonlcon1},
       {'type':'ineq','fun':nonlcon2})

res = minimize(fun = fun1, x0 = x0,
            constraints = con,
            bounds=None,tol=None,options=None,args=())


print("默认算法（类似内点法等情况）求解结果：")
print("最优解：",res.x)
print("最优值：",res.fun)