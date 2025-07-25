import matplotlib.pyplot as plt
import numpy as np
#参数初始化：投放10000个点，圆半径为1，圆心坐标（1，1）
p = 100000   #总共要投放的点数
r = 1   #圆的半径
x0,y0 = 1,1 #圆心的坐标
n = 0   #初始时还未投放点，有0个点在圆内


plt.figure()
plt.title('Monte Carlo Simulation for Estimating Pi')
plt.xlabel('x')
plt.ylabel('y')
#保持绘图窗口，多次绘图
for i in range(p): # 对于要投放的总共p个点
 #np.random.rand()函数产生在（0，1)之间的随机数
    px=np.random.rand()*2 #随机生成该点的横坐标
    py=np.random.rand()*2 #随机生成该点的纵坐标
                #判断点是否在圆内
    if(px-x0) ** 2+(py-y0) ** 2<r ** 2:  #横纵坐标的平方和小于半径，则在圆内
         plt.plot(px,py,'.',color='b')   #圆内点用蓝色表示
         n+=1
    else:
        plt.plot(px,py,'.',color='r')#圆外点用红色表示
    
plt.axis('equal')#绘图时横纵坐标单位长度相同，便于观察圆
plt.show()

# 计算π
s = (n/p)*4
print('π=',s)