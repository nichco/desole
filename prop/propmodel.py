import csdl
import python_csdl_backend
import numpy as np
import pickle
from smt.surrogate_models import KRG




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



# create the training data:
nrpm = np.linspace(500,5000,6) # rotor speed (rpm)
vaxial = np.linspace(0,100,6) # axial inflow (m/s)
vtan = np.linspace(0,100,6) # edgewise inflow (m/s)
x = np.zeros([216,3])
index = 0
for i, rpm in enumerate(nrpm):
    for j, u in enumerate(vaxial):
        for k, v in enumerate(vtan):
            x[index,0] = nrpm[i]
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
        ct, cp = csdl.custom(rpm, vAxial, vTan, op=PropExplicit(num_nodes=num))
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
        self.add_input('vTan', shape=num)

        # output: thrust coefficient and power coefficient
        self.add_output('ct', shape=num)
        self.add_output('cp', shape=num)

        # declare derivatives
        self.declare_derivatives('ct', 'rpm')
        self.declare_derivatives('ct', 'vAxial')
        self.declare_derivatives('ct', 'vTan')
        self.declare_derivatives('cp', 'rpm')
        self.declare_derivatives('cp', 'vAxial')
        self.declare_derivatives('cp', 'vTan')

    def compute(self, inputs, outputs):
        num = self.parameters['num_nodes']

        # the surrogate model interpolation:
        point = np.zeros([num,3])
        point[:,0] = inputs['rpm']
        point[:,1] = inputs['vAxial']
        point[:,2] = inputs['vTan']
        ct = sm_ct.predict_values(point)
        cp = sm_cp.predict_values(point)

        # define the outputs:
        outputs['ct'] = 1*ct
        outputs['cp'] = 1*cp

    def compute_derivatives(self, inputs, derivatives):
        num = self.parameters['num_nodes']

        point = np.zeros([num,3])
        point[:,0] = inputs['rpm']
        point[:,1] = inputs['vAxial']
        point[:,2] = inputs['vTan']
        dct_drpm = sm_ct.predict_derivatives(point, 0)
        dct_dvaxial = sm_ct.predict_derivatives(point, 1)
        dct_dvtan = sm_ct.predict_derivatives(point, 2)
        dcp_drpm = sm_cp.predict_derivatives(point, 0)
        dcp_dvaxial = sm_cp.predict_derivatives(point, 1)
        dcp_dvtan = sm_cp.predict_derivatives(point, 2)

        derivatives['ct', 'rpm'] = np.diag(dct_drpm.flatten())
        derivatives['ct', 'vAxial'] = np.diag(dct_dvaxial.flatten())
        derivatives['ct', 'vTan'] = np.diag(dct_dvtan.flatten())
        derivatives['cp', 'rpm'] = np.diag(dcp_drpm.flatten())
        derivatives['cp', 'vAxial'] = np.diag(dcp_dvaxial.flatten())
        derivatives['cp', 'vTan'] = np.diag(dcp_dvtan.flatten())




if __name__ == '__main__':
    
    name = 'lift'
    sim = python_csdl_backend.Simulator(Prop(name=name, num_nodes=4, d=2.4))
    sim.run()

    print('C_T: ', sim[name + '_C_T'])
    print('C_P: ', sim[name + '_C_P'])
    print('Thrust: ', sim[name + '_thrust'])
    print('Power: ', sim[name + '_power'])

    sim.check_partials(step=1E-6, compact_print=True)
    