import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from smt.surrogate_models import KRG, RBF
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({'font.size': 12})



efficiency = np.loadtxt('efficiency.txt').astype(np.double)
speed = np.loadtxt('speed.txt').astype(np.double)
torque = np.loadtxt('torque.txt').astype(np.double)


n = len(efficiency[:,0]) # 16
m = len(efficiency[0,:]) # 16


xt = np.zeros((n*m + 4, 2))
yt = np.zeros((n*m + 4))
index = 0
for i in range(n):
    for j in range(m):
        xt[index, 0] = speed[i,j]
        xt[index, 1] = torque[i,j]
        yt[index] = efficiency[i,j] - 0.1
        
        index += 1


xt[-1,0] = 6000
xt[-1,1] = 1100
yt[-1] = 0

xt[-2,0] = 4000
xt[-2,1] = 1500
yt[-2] = -0.1

xt[-3,0] = 2500
xt[-3,1] = 2000
yt[-3] = -0.4

xt[-4,0] = 3000
xt[-4,1] = 1700
yt[-4] = -0.1


levels = np.arange(0.0, 1.1, 0.03)
plt.scatter(speed, torque, efficiency, zorder=10, color='k')
plt.contourf(speed, torque, efficiency, levels=levels)
plt.show()


#sm = KRG(theta0=[1e-4], print_global=False, print_solver=False, hyper_opt='TNC')
sm = RBF(d0=200000,print_global=False,print_solver=False,)
sm.set_training_values(xt, yt)
sm.train()





num = 100
rpm = np.linspace(0,6000,num)
q = np.linspace(0,2000,num)
data = np.zeros((num,num))

for i, r in enumerate(rpm):
    for j, t in enumerate(q):
        point = np.zeros([1, 2])
        point[0][0] = r
        point[0][1] = t

        ans = sm.predict_values(point)
        data[i,j] = ans





levels = np.arange(0.0, 1, 0.03)
plt.contourf(rpm,q,data,levels=levels)
plt.colorbar(shrink=1)
plt.show()







