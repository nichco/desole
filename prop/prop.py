import numpy as np
import pickle
from smt.surrogate_models import KRG
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


ctfile = open('ct.pkl', 'rb')
datact = pickle.load(ctfile)
cpfile = open('cp.pkl', 'rb')
datacp = pickle.load(cpfile)

n = np.linspace(500,5000,6) # rotor speed (rpm)
vaxial = np.linspace(0,100,6) # axial inflow (m/s)
vtan = np.linspace(0,100,6) # edgewise inflow (m/s)

x = np.zeros([216,3])
index = 0
for i, rpm in enumerate(n):
    for j, u in enumerate(vaxial):
        for k, v in enumerate(vtan):
            x[index,0] = n[i]
            x[index,1] = vaxial[j]
            x[index,2] = vtan[k]
            index += 1


y = np.reshape(datact, (216, 1))


sm = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm.set_training_values(x, y)
sm.train()



"""
point = np.zeros([1, 3])
point[0][0] = 1500
point[0][1] = 0
point[0][2] = 0

ct = sm.predict_values(point)
print(ct)
"""
num = 100
n = np.linspace(500,5000,num) # rotor speed (rpm)
vaxial = np.linspace(0,100,num) # axial inflow (m/s)
vtan = np.linspace(0,100,num) # edgewise inflow (m/s)
data = np.zeros((num,num))

for i, u in enumerate(vaxial):
    for j, v in enumerate(vtan):
        point = np.zeros([1, 3])
        point[0][0] = 1000
        point[0][1] = u
        point[0][2] = v

        ct = sm.predict_values(point)
        data[i,j] = ct


plt.contourf(vaxial,vtan,data)
plt.colorbar(shrink=1)
plt.show()