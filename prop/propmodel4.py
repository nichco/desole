import csdl
import python_csdl_backend
import numpy as np
import pickle
from smt.surrogate_models import KRG
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)



# import the data:
ctfile = open('prop/ctbem.pkl', 'rb')
datact_in = pickle.load(ctfile)
cpfile = open('prop/cpbem.pkl', 'rb')
datacp_in = pickle.load(cpfile)

datact = np.zeros((10,10))
datacp = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        if datacp_in[i,j] < 1E-2:
            datacp[i,j] = 0
        else:
            datacp[i,j] = datacp_in[i,j]
        if datact_in[i,j] < 1E-2:
            datact[i,j] = 0
        else:
            datact[i,j] = datact_in[i,j]



# create the training data:
rpm = np.linspace(500,5000,10) # rotor speed (rpm)
vaxial = np.linspace(0,100,10) # axial inflow (m/s)
x = np.zeros((100,2))
yct = np.zeros((100))
ycp = np.zeros((100))
index = 0
for i, n in enumerate(rpm):
    for j, u in enumerate(vaxial):
        x[index,0] = 1*n
        x[index,1] = 1*u
        yct[index] = datact[i,j]
        ycp[index] = datacp[i,j]
        index += 1

# train the model:
sm_ct = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_ct.set_training_values(x, yct)
sm_ct.train()

sm_cp = KRG(theta0=[1e-2], print_global=False, print_solver=False, hyper_opt='TNC')
sm_cp.set_training_values(x, ycp)
sm_cp.train()



# num = 100
# rpm = np.linspace(500,5000,num)
# vaxial = np.linspace(0,100,num)
# datact = np.zeros((num,num))
# datacp = np.zeros((num,num))
# for i, n in enumerate(rpm):
#     for j, u in enumerate(vaxial):
#         point = np.zeros([1, 2])
#         point[0][0] = 1*n
#         point[0][1] = 1*u
#         ct = sm_ct.predict_values(point)
#         cp = sm_cp.predict_values(point)
#         datact[i,j] = ct
#         datacp[i,j] = cp

# plt.contourf(rpm,vaxial,np.transpose(datact))
# plt.colorbar(shrink=1)
# plt.show()

# plt.contourf(rpm,vaxial,np.transpose(datacp))
# plt.colorbar(shrink=1)
# plt.show()

# exit()

class Prop(csdl.Model):
    def initialize(self):
        self.parameters.declare('name',types=str)
        self.parameters.declare('num_nodes')
        self.parameters.declare('d')

    def define(self):
        num = self.parameters['num_nodes']
        name = self.parameters['name']
        d = self.parameters['d']

        rpm_in = self.declare_variable(name + '_rpm', shape=num, val=1500)
        vaxial_in = self.declare_variable(name + '_vaxial', shape=num, val=10)

        rpm = self.register_output('rpm', 1*rpm_in)
        vAxial = self.register_output('vAxial', 1*vaxial_in)

        # custom operation insertion
        ct, cp = csdl.custom(rpm, vAxial, op=PropExplicit(num_nodes=num))
        C_T = self.register_output(name + '_C_T', 1*ct)
        C_P = self.register_output(name + '_C_P', 1*cp)

        rho = self.declare_variable('density', shape=(num), val=1.225)
        n = rpm/60
        self.register_output(name + '_thrust', C_T*rho*(n**2)*(d**4))
        self.register_output(name + '_power', C_P*rho*(n**3)*(d**5))




class PropExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')

        
    def define(self):
        num = self.parameters['num_nodes']

        # inputs:
        self.add_input('rpm', shape=num)
        self.add_input('vAxial', shape=num)

        # output: thrust coefficient and power coefficient
        self.add_output('ct', shape=num)
        self.add_output('cp', shape=num)

        # declare derivatives
        self.declare_derivatives('ct', 'rpm')
        self.declare_derivatives('ct', 'vAxial')
        self.declare_derivatives('cp', 'rpm')
        self.declare_derivatives('cp', 'vAxial')

    def compute(self, inputs, outputs):
        num = self.parameters['num_nodes']

        # the surrogate model interpolation:
        point = np.zeros([num,2])
        point[:,0] = inputs['rpm']
        point[:,1] = inputs['vAxial']
        ct = sm_ct.predict_values(point)
        cp = sm_cp.predict_values(point)

        # define the outputs:
        outputs['ct'] = 1*ct
        outputs['cp'] = 1*cp

    def compute_derivatives(self, inputs, derivatives):
        num = self.parameters['num_nodes']

        point = np.zeros([num,2])
        point[:,0] = inputs['rpm']
        point[:,1] = inputs['vAxial']
        dct_drpm = sm_ct.predict_derivatives(point, 0)
        dct_dvaxial = sm_ct.predict_derivatives(point, 1)
        dcp_drpm = sm_cp.predict_derivatives(point, 0)
        dcp_dvaxial = sm_cp.predict_derivatives(point, 1)


        derivatives['ct', 'rpm'] = np.diag(dct_drpm.flatten())
        derivatives['ct', 'vAxial'] = np.diag(dct_dvaxial.flatten())
        derivatives['cp', 'rpm'] = np.diag(dcp_drpm.flatten())
        derivatives['cp', 'vAxial'] = np.diag(dcp_dvaxial.flatten())



if __name__ == '__main__':
    
    name = 'lift'
    sim = python_csdl_backend.Simulator(Prop(name=name, num_nodes=4, d=2.4))
    sim.run()

    print('C_T: ', sim[name + '_C_T'])
    print('C_P: ', sim[name + '_C_P'])
    print('Thrust: ', sim[name + '_thrust'])
    print('Power: ', sim[name + '_power'])

    sim.check_partials(step=1E-6, compact_print=True)
