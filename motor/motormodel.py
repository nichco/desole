from motor.motor import sm_eta
import csdl
import python_csdl_backend
import numpy as np




class Motor(csdl.Model):
    def initialize(self):
        self.parameters.declare('name',types=str)
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']
        name = self.parameters['name']

        q = self.declare_variable(name + '_torque', shape=n)
        r = self.declare_variable(name + 'm', shape=n)

        # custom operation insertion
        eta = csdl.custom(q, r, op=MotorExplicit(name=name,num_nodes=n))

        self.register_output(name + '_eta', 1*eta)
        
        
        
class MotorExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('name',types=str)
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']
        name = self.parameters['name']

        # input: motor torque and rpm
        self.add_input(name+'_torque', shape=n)
        self.add_input(name+'m', shape=n)

        # output: efficiency
        self.add_output(name+'eta', shape=n)

        # declare derivatives
        self.declare_derivatives(name+'eta', name+'_torque')
        self.declare_derivatives(name+'eta', name+'m')
        
        
    def compute(self, inputs, outputs):
        n = self.parameters['num_nodes']
        name = self.parameters['name']

        eta = np.zeros((n))
        for i in range(n):
            point = np.flip(np.array([[inputs[name+'_torque'][i], inputs[name+'m'][i]]]).reshape(1,2))
            eta[i] = sm_eta.predict_values(point)

        # define outputs
        outputs[name+'eta'] = 1*eta
        
        
    def compute_derivatives(self, inputs, derivatives):
        n = self.parameters['num_nodes']
        name = self.parameters['name']

        deta_dq = np.zeros((n))
        deta_dm = np.zeros((n))

        for i in range(n):
            point = np.flip(np.array([[inputs[name+'_torque'][i], inputs[name+'m'][i]]]).reshape(1,2))
            deta_dq[i] = sm_eta.predict_derivatives(point, 1)
            deta_dm[i] = sm_eta.predict_derivatives(point, 0)
        
        derivatives[name+'eta', name+'_torque'] = np.diag(deta_dq)
        derivatives[name+'eta', name+'m'] = np.diag(deta_dm)
        



if __name__ == '__main__':
    name = 'cruise'
    sim = python_csdl_backend.Simulator(Motor(name=name,num_nodes=5))
    sim['cruise_torque'] = 2000
    sim['cruise_m'] = 200
    sim.run()

    eta = sim[name+'eta']
    print('efficiency: ',eta)

    # print partials
    sim.check_partials(compact_print=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        