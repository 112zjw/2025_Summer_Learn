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

#使用蒙特卡罗的方法来找初始值（推荐）
n=10000000
#np.random.uniform是NumPy库中用于生成在指定区间内均匀分布的随机数（或随机数组）的函数。
#用法同np.random.randint，不过这里生成的是随机数，不是整数
x1 = np.random.uniform(-100,100, size=n)
x2 = np.random.uniform(-100,100, size=n)

fmin=100000000
x0 = None

for i in range(n):
    x = np.array([x1[i],x2[i]])
#后续的约束条件判断和结果处理逻辑保持不变
    if nonlcon1(x) >= 0 and nonlcon2(x) >= 0:
        result = fun1(x)
        if result < fmin:
            fmin = result
            x0 = x
print("蒙特卡罗选取的初始值为：",x0)


con = ({'type':'ineq','fun':nonlcon1},
       {'type':'ineq','fun':nonlcon2})

res = minimize(fun = fun1, x0 = x0,
            constraints = con,
            bounds=None,tol=None,options=None,args=())

print("默认算法（类似内点法）求解结果：")
print("最优解：",res.x)
print("最优值：",res.fun)

res = minimize(fun = fun1, x0 = x0, method = 'SLSQP',
            constraints = con,
            bounds=None,tol=None,options=None,args=())

print("算法（序列二次规划）求解结果：")
print("最优解：",res.x)
print("最优值：",res.fun)