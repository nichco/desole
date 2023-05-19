import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


n = 3
dt = 1
data = np.random.rand(n)
num = 5


y = np.zeros((num*(n - 1)))
d = 1/num

index = 0
for i in range(n - 1):
    m = (data[i+1] - data[i])
    for j in range(num):
        x = j*d
        y[index] = m*x + data[i]

        index += 1

    





x = np.linspace(0,n,n)
xn = np.linspace(0,n,num*())
plt.scatter(x, data, color='goldenrod', s=60)
plt.scatter(xn, y, color='k', s=30)
#plt.plot(x, data)
plt.show()
