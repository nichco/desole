import numpy as np
import pickle
from smt.surrogate_models import KRG
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# import the data:
ctfile = open('prop/ct.pkl', 'rb')
datact_in = pickle.load(ctfile)
cpfile = open('prop/cp.pkl', 'rb')
datacp_in = pickle.load(cpfile)

datact = np.zeros((6,6,6))
datacp = np.zeros((6,6,6))
for i in range(6):
    for j in range(6):
        for k in range(6):
            if datacp_in[i,j,k] < 1E-2:
                datacp[i,j,k] = 0
            else:
                datacp[i,j,k] = datacp_in[i,j,k]
                
            if datact_in[i,j,k] < 1E-2:
                datact[i,j,k] = 0
            else:
                datact[i,j,k] = datact_in[i,j,k]


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


yct = np.reshape(datact, (216, 1))
ycp = np.reshape(datacp, (216, 1))

# train the model:
sm_ct = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_ct.set_training_values(x, yct)
sm_ct.train()
#self.sm_ct = sm_ct

sm_cp = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_cp.set_training_values(x, ycp)
sm_cp.train()
#self.sm_cp = sm_cp



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
datact = np.zeros((num,num))
datacp = np.zeros((num,num))

for i, u in enumerate(vaxial):
    for j, v in enumerate(vtan):
        point = np.zeros([1, 3])
        point[0][0] = 1000
        point[0][1] = u
        point[0][2] = v

        ct = sm_ct.predict_values(point)
        cp = sm_cp.predict_values(point)
        datact[i,j] = ct
        datacp[i,j] = cp


plt.contourf(vaxial,vtan,datact)
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datacp)
plt.colorbar(shrink=1)
plt.show()