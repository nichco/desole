import csdl
import python_csdl_backend
import numpy as np
import pickle
from smt.surrogate_models import KRG




# import the data:
ctfile = open('prop/ctbem.pkl', 'rb')
datact = pickle.load(ctfile)
cpfile = open('prop/cpbem.pkl', 'rb')
datacp = pickle.load(cpfile)

# create the training data:
nrpm = np.linspace(500,5000,10) # rotor speed (rpm)
vaxial = np.linspace(0,100,10) # axial inflow (m/s)
x = np.zeros([216,3])
index = 0
for i, rpm in enumerate(nrpm):
    for j, u in enumerate(vaxial):
        x[index,0] = rpm
        x[index,1] = u
        index += 1

yct = np.reshape(datact, (100, 1))
ycp = np.reshape(datacp, (100, 1))

# train the model:
sm_ct = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_ct.set_training_values(x, yct)
sm_ct.train()

sm_cp = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_cp.set_training_values(x, ycp)
sm_cp.train()