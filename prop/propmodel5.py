import numpy as np
import csdl
from smt.surrogate_models import RMTB
import python_csdl_backend
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

ctarr = np.array([[ 0.50845263,  0.48114841,  0.46048069,  0.45263168,  0.44918747,  0.45263168, 0.46048069,  0.48114841,  0.50845263],
    [ 0.41469071,  0.37799377,  0.35409169,  0.33544928,  0.32626682,  0.33544928, 0.35409169,  0.37799377,  0.41469071],
    [ 0.35626874,  0.31375178,  0.27994125,  0.25443275,  0.24395818,  0.25443275, 0.27994125,  0.31375178,  0.35626874],
    [ 0.33795284,  0.29184094,  0.25599661,  0.2345676,   0.26461192,  0.2345676, 0.25599661,  0.29184094,  0.33795284],
    [ 0.34362491,  0.29646818,  0.26162671,  0.24395124,  0.24127203,  0.24395124, 0.26162671,  0.29646818,  0.34362491],
    [ 0.34468518,  0.29754716,  0.26353205,  0.24336877,  0.23664139,  0.24336877, 0.26353205,  0.29754716,  0.34468518],
    [ 0.31969465,  0.25751606,  0.21456789,  0.20389248,  0.20295964,  0.20389248, 0.21456789,  0.25751606,  0.31969465],
    [ 0.28051414,  0.20418248,  0.13734402,  0.10851712,  0.11667478,  0.10851712, 0.13734402,  0.20418248,  0.28051414],
    [ 0.23358927,  0.14279756,  0.0603329,   0.00172831, -0.0176337,   0.00172831, 0.0603329,   0.14279756,  0.23358927]])

cparr = np.array([[0.501836,   0.46373982, 0.43129937, 0.41990135, 0.41816146, 0.41990135, 0.43129937, 0.46373982, 0.501836  ],
    [0.41321428, 0.37064409, 0.33877887, 0.31762158, 0.30942568, 0.31762158, 0.33877887, 0.37064409, 0.41321428],
    [0.33611122, 0.29281586, 0.25239685, 0.22285395, 0.21143836, 0.22285395, 0.25239685, 0.29281586, 0.33611122],
    [0.28741357, 0.23773751, 0.19594548, 0.17230629, 0.24126226, 0.17230629, 0.19594548, 0.23773751, 0.28741357],
    [0.28558827, 0.23334922, 0.19459788, 0.17535239, 0.170538,   0.17535239, 0.19459788, 0.23334922, 0.28558827],
    [0.31958289, 0.27173158, 0.23914474, 0.22150391, 0.2156555,  0.22150391, 0.23914474, 0.27173158, 0.31958289],
    [0.32641025, 0.26698576, 0.24129197, 0.25094166, 0.2501908,  0.25094166, 0.24129197, 0.26698576, 0.32641025],
    [0.29057492, 0.21326359, 0.15586953, 0.1693413,  0.19980031, 0.1693413, 0.15586953, 0.21326359, 0.29057492],
    [0.21951724, 0.12096531, 0.03968045, 0.01119779, 0.01379037, 0.01119779, 0.03968045, 0.12096531, 0.21951724]])


# construct training data in a form smt can use
N = 9
xta = np.linspace(-100,100,N)
xtb = np.linspace(-100,100,N)
xt = np.zeros((N*N,2))
index = 0
for i in range(N):
    for j in range(N):
        xt[index,:] = [xta[i],xtb[j]]
        index += 1

yt_ct = np.zeros((N*N,1))
index = 0
for i in range(N):
    for j in range(N):
        yt_ct[index,:] = ctarr[i,j]
        index += 1

yt_cp = np.zeros((N*N,1))
index = 0
for i in range(N):
    for j in range(N):
        yt_cp[index,:] = cparr[i,j]
        index += 1

xlimits = np.array([[-100.0, 100.0], [-100.0, 100.0]])

# construct surrogate model
sm_ct = RMTB(
            xlimits=xlimits,
            order=4,
            num_ctrl_pts=4,
            energy_weight=1e-10,
            regularization_weight=0.0,
            print_global=False,
            print_solver=False,)

sm_cp = RMTB(
            xlimits=xlimits,
            order=3,
            num_ctrl_pts=3, # 4 cp's and order 4 introduces local minimum
            energy_weight=1e-10,
            regularization_weight=0.0,
            print_global=False,
            print_solver=False,)

# train model
sm_ct.set_training_values(xt, yt_ct)
sm_ct.train()

sm_cp.set_training_values(xt, yt_cp)
sm_cp.train()




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
        vtan_in = self.declare_variable(name + '_vtan', shape=num, val=10)

        rpm = self.register_output('rpm', 1*rpm_in)
        vAxial = self.register_output('vAxial', 1*vaxial_in)
        vTan = self.register_output('vTan', 1*vtan_in)

        # custom operation insertion
        ct, cp = csdl.custom(vAxial, vTan, op=PropExplicit(num_nodes=num))
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
        n = self.parameters['num_nodes']

        # input: axial and tangential freestream velocities
        self.add_input('vAxial', shape=n)
        self.add_input('vTan', shape=n)

        # output: thrust coefficient and power coefficient
        self.add_output('ct', shape=n)
        self.add_output('cp', shape=n)

        # declare derivatives
        self.declare_derivatives('ct', 'vAxial')
        self.declare_derivatives('ct', 'vTan')
        self.declare_derivatives('cp', 'vAxial')
        self.declare_derivatives('cp', 'vTan')

    def compute(self, inputs, outputs):
        n = self.parameters['num_nodes']

        # surrogate model interpolation
        # point = np.array([[inputs[name+'vAxial'], inputs[name+'vTan']]]).reshape(1,2)
        # ct = sm_ct.predict_values(point)
        # cp = sm_cp.predict_values(point)
        ct = np.zeros((n))
        cp = np.zeros((n))
        for i in range(n):
            point = np.array([[inputs['vAxial'][i], inputs['vTan'][i]]]).reshape(1,2)
            ct[i] = sm_ct.predict_values(point)
            cp[i] = sm_cp.predict_values(point)

        # define outputs
        outputs['ct'] = 1*ct
        outputs['cp'] = 1*cp

    def compute_derivatives(self, inputs, derivatives):
        n = self.parameters['num_nodes']
        """
        # compute derivatives
        point = np.array([[inputs[name+'vAxial'], inputs[name+'vTan']]]).reshape(1,2)
        dct_dvaxial = sm_ct.predict_derivatives(point, 0)
        dct_dvtan = sm_ct.predict_derivatives(point, 0)
        dcp_dvaxial = sm_cp.predict_derivatives(point, 0)
        dcp_dvtan = sm_cp.predict_derivatives(point, 0)

        # assign derivatives
        derivatives[name+'ct', name+'vAxial'] = dct_dvaxial
        derivatives[name+'ct', name+'vTan'] = dct_dvtan
        derivatives[name+'cp', name+'vAxial'] = dcp_dvaxial
        derivatives[name+'cp', name+'vTan'] = dcp_dvtan
        """
        dct_dvaxial = np.zeros((n))
        dct_dvtan = np.zeros((n))
        dcp_dvaxial = np.zeros((n))
        dcp_dvtan = np.zeros((n))
        for i in range(n):
            point = np.array([[inputs['vAxial'][i], inputs['vTan'][i]]]).reshape(1,2)
            dct_dvaxial[i] = sm_ct.predict_derivatives(point, 0)
            dct_dvtan[i] = sm_ct.predict_derivatives(point, 1)
            dcp_dvaxial[i] = sm_cp.predict_derivatives(point, 0)
            dcp_dvtan[i] = sm_cp.predict_derivatives(point, 1)

        derivatives['ct', 'vAxial'] = np.diag(dct_dvaxial)
        derivatives['ct', 'vTan'] = np.diag(dct_dvtan)
        derivatives['cp', 'vAxial'] = np.diag(dcp_dvaxial)
        derivatives['cp', 'vTan'] = np.diag(dcp_dvtan)





if __name__ == '__main__':
    
    name = 'lift'
    sim = python_csdl_backend.Simulator(Prop(name=name, num_nodes=4, d=2.4))
    sim.run()

    print('C_T: ', sim[name + '_C_T'])
    print('C_P: ', sim[name + '_C_P'])
    print('Thrust: ', sim[name + '_thrust'])
    print('Power: ', sim[name + '_power'])

    sim.check_partials(step=1E-6, compact_print=True)