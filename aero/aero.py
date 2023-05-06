import numpy as np
from smt.surrogate_models import RBF
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


alpha = np.deg2rad(np.array([-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,]))
cl05 = np.array([-0.01,-0.42,-0.75,-0.95,-1.08,-1.17,-1.25,-1.3,-1.33,-1.34,-1.32,-1.27,-1.24,-1.29,-1.28,-1.07869,-0.67686,-0.27838,0.12907,0.5325,
                 0.92638,1.28286,1.35463,1.34,1.23,1.18,1.22,1.28,1.32,1.34,1.33,1.3,1.25,1.17,1.08,0.95,0.75,0.42,0.01,])
cl1 = np.array([-0.01,-0.42,-0.75,-0.95,-1.08,-1.17,-1.25,-1.3,-1.33,-1.34,-1.32,-1.27,-1.24,-1.29,-1.28,-1.08261,-0.67929,-0.27926,0.13037,0.53477,0.92992,
                1.2841,1.35493,1.34,1.23,1.18,1.22,1.28,1.32,1.34,1.33,1.3,1.25,1.17,1.08,0.95,0.75,0.42,0.01,])
cl2 = np.array([-0.01,-0.42,-0.75,-0.95,-1.08,-1.17,-1.25,-1.3,-1.33,-1.34,-1.32,-1.27,-1.24,-1.29,-1.28,-1.09877,-0.68933,-0.28345,0.13125,0.54489,
                0.94455,1.28921,1.35625,1.34,1.23,1.18,1.22,1.28,1.32,1.34,1.33,1.3,1.25,1.17,1.08,0.95,0.75,0.42,0.01,])
cl3 = np.array([-0.01,-0.42,-0.75,-0.95,-1.08,-1.17,-1.25,-1.3,-1.33,-1.34,-1.32,-1.27,-1.24,-1.29,-1.28,-1.12766,-0.70729,-0.29033,0.13461,0.55804,
                0.96982,1.29469,1.35836,1.34,1.23,1.18,1.22,1.28,1.32,1.34,1.33,1.3,1.25,1.17,1.08,0.95,0.75,0.42,0.01,])



# create the training data:
mach = np.array([0.05, 0.1, 0.2, 0.3])
x = np.zeros((len(alpha)*len(mach), 2))
cldata = np.zeros((len(alpha)*len(mach)))
index = 0
for i, a in enumerate(alpha):
    for j, m in enumerate(mach):
        x[index,0] = a
        x[index,1] = m
        if j == 0: cldata[index] = cl05[i]
        elif j == 1: cldata[index] = cl1[i]
        elif j == 2: cldata[index] = cl2[i]
        elif j == 3: cldata[index] = cl3[i]
        index += 1



#sm = KRG(theta0=[1e-2], print_global=False, print_solver=False)
sm = RBF(d0=0.3,print_global=False,print_solver=False,)
sm.set_training_values(x, cldata)
sm.train()



num = 100
aoa = np.deg2rad(np.linspace(-90,90,num))
machnum = np.linspace(0.05,0.3,num)
"""
data = np.zeros((num,num))
for i, a in enumerate(aoa):
    for j, m in enumerate(machnum):
        point = np.zeros([1, 2])
        point[0][0] = a
        point[0][1] = m

        cl = sm.predict_values(point)
        data[i,j] = cl

plt.contourf(aoa,machnum, data)
plt.colorbar(shrink=1)
plt.show()
"""

for i, m in enumerate(machnum):
    data = np.zeros((len(aoa)))
    ddata = np.zeros((len(aoa)))
    for j, a in enumerate(aoa):
        point = np.zeros([1, 2])
        point[0][0] = a
        point[0][1] = m

        cl = sm.predict_values(point)
        data[j] = cl

        dcl_da = sm.predict_derivatives(point, 0)
        ddata[j] = dcl_da

    plt.plot(data)
    plt.plot(ddata)

plt.show()



"""
data = np.zeros((1,num))
for i, a in enumerate(aoa):
    point = np.array([a])
    cl = sm.predict_values(point)
    data[0,i] = cl

plt.plot(aoa,data[0,:])
plt.show()
"""