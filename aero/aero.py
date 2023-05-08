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
cd05 = np.array([1.85,1.84,1.83,1.81,1.78,1.73,1.67,1.59,1.48,1.35,1.2,0.95,0.57,0.27,0.10316,0.04962,0.02675,0.01523,0.01588,0.02745,0.05054,0.09198,
                 0.20645,0.32825,0.5,0.73,0.93,1.125,1.29,1.43,1.54,1.63,1.7,1.76,1.81,1.85,1.87,1.89,1.9,])
cd1 = np.array([1.85,1.84,1.83,1.81,1.78,1.73,1.67,1.59,1.48,1.35,1.2,0.95,0.58,0.28,0.1024,0.04765,0.02455,0.0129,0.01358,0.02516,0.0484,0.09098,0.20628,
                0.32875,0.51,0.74,0.935,1.13,1.29,1.43,1.54,1.63,1.7,1.76,1.81,1.85,1.87,1.89,1.9,])
cd2 = np.array([1.85,1.84,1.83,1.81,1.78,1.73,1.67,1.59,1.48,1.35,1.2,0.95,0.59,0.29,0.10651,0.04689,0.02297,0.01107,0.01169,0.02346,0.04729,
                0.09449,0.21321,0.33818,0.52,0.75,0.94,1.135,1.295,1.43,1.54,1.63,1.7,1.76,1.81,1.85,1.87,1.89,1.9,])
cd3 = np.array([1.85,1.84,1.83,1.81,1.78,1.73,1.67,1.59,1.48,1.35,1.2,0.95,0.6,0.3,0.11651,0.04797,0.02267,0.00984,0.01059,0.02314,0.04781,0.10372,
                0.22801,0.35719,0.54,0.76,0.96,1.14,1.3,1.43,1.54,1.63,1.7,1.76,1.81,1.85,1.87,1.89,1.9,])



# create the training data:
mach = np.array([0.05, 0.1, 0.2, 0.3])
x = np.zeros((len(alpha)*len(mach), 2))
cldata = np.zeros((len(alpha)*len(mach)))
cddata = np.zeros((len(alpha)*len(mach)))
index = 0
for i, a in enumerate(alpha):
    for j, m in enumerate(mach):
        x[index,0] = a
        x[index,1] = m
        if j == 0: 
            cldata[index] = cl05[i]
            cddata[index] = cd05[i]
        elif j == 1: 
            cldata[index] = cl1[i]
            cddata[index] = cd1[i]
        elif j == 2: 
            cldata[index] = cl2[i]
            cddata[index] = cd2[i]
        elif j == 3: 
            cldata[index] = cl3[i]
            cddata[index] = cd3[i]
        index += 1



sm_cl = RBF(d0=0.3,print_global=False,print_solver=False,)
sm_cl.set_training_values(x, cldata)
sm_cl.train()

sm_cd = RBF(d0=0.5,print_global=False,print_solver=False,)
sm_cd.set_training_values(x, cddata + 0.03)
sm_cd.train()



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
    datacl = np.zeros((len(aoa)))
    datacd = np.zeros((len(aoa)))
    for j, a in enumerate(aoa):
        point = np.zeros([1, 2])
        point[0][0] = a
        point[0][1] = m

        cl = sm_cl.predict_values(point)
        datacl[j] = cl

        cd = sm_cd.predict_values(point)
        datacd[j] = cd

    plt.plot(aoa,datacl)
    #plt.plot(aoa,datacd)
    #plt.scatter(alpha,cd1)
    #plt.plot(ddata)

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