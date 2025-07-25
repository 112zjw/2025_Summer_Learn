import numpy as np

n = 100000
a = 0 # a表示不改变主意能赢得汽车的次数
b = 0 # b表示改变主意能赢得汽车的次数

for i in range(n):
    x = np.random.randint(1,4)  # 汽车所在
    y = np.random.randint(1,4)  # 第一次选择
    
    if x == y :
        a += 1
    else :
        b += 1    

print("不改变主意的获奖概率：", a/n)
print("改变主意的获奖概率：", b/n)