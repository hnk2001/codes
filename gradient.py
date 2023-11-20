x = 2
lr = 0.01
prev_step_size = 1
iters = 0
max_iter =10000
precision = 0.000001

gf= lambda x : (x+3)**2

gd= []
import matplotlib.pyplot as plt
while precision<prev_step_size and iters<max_iter:
    prev = x
    iters+=1
    x = x - lr*gf(prev)
    prev = abs(x-prev)
    print("iteration", iters, "x: ",x )
    gd.append(x)

print("local minima:- ",x)
plt.plot(gd)
plt.show()