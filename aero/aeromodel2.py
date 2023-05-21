import csdl
import python_csdl_backend
import numpy as np
from smt.surrogate_models import RBF
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


xt = np.deg2rad(np.array([-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,]))
cldata = np.array([-0.01,-0.35,-0.65,-0.85,-1,-1.12,-1.18,-1.22,-1.235,-1.24,-1.22,-1.19,-1.2,-1.29,-1.28,-1.07869,-0.67686,-0.27838,0.12907,0.5325,0.92638,1.28286,1.35463,
1.34,1.23,1.18,1.19,1.24,1.28,1.29,1.28,1.265,1.22,1.14,1.02,0.85,0.64,0.35,0.01,])
cddata = np.array([1.85,1.84,1.83,1.81,1.78,1.73,1.67,1.59,1.48,1.35,1.2,0.95,0.58,0.28,0.1024,0.04765,0.02455,0.0129,0.01358,0.02516,0.0484,0.09098,0.20628,
                0.32875,0.51,0.74,0.935,1.13,1.29,1.43,1.54,1.63,1.7,1.76,1.81,1.85,1.87,1.89,1.9,])




class Aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('wing_area')

    def define(self):
        num = self.parameters['num_nodes']
        s = self.parameters['wing_area']

        alpha = self.declare_variable('alpha', shape=(num), val=0.01)
        v = self.declare_variable('v', shape=num)
        rho = self.declare_variable('density', shape=(num), val=1.225)

        # custom operation insertion:
        cl, cd = csdl.custom(alpha, op=AeroExplicit(num_nodes=num))
        C_L = self.register_output('C_L', 1*cl)
        C_D = self.register_output('C_D', 1*cd)

        lift = 0.5*rho*(v**2)*s*C_L
        drag = 0.5*rho*(v**2)*s*C_D
        self.register_output('lift', lift)
        self.register_output('drag', drag)

        #self.print_var(lift)
        #self.print_var(drag)



class AeroExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')

        # create the training data:
        sm_cl = RBF(d0=0.3,print_global=False,print_solver=False,)
        sm_cl.set_training_values(xt, cldata)
        sm_cl.train()
        self.sm_cl = sm_cl

        sm_cd = RBF(d0=0.6,print_global=False,print_solver=False,)
        sm_cd.set_training_values(xt, cddata)
        sm_cd.train()
        self.sm_cd = sm_cd

        # xe = np.deg2rad(np.linspace(-90,90,100))
        # cl_eval = sm_cl.predict_values(xe)
        # cd_eval = sm_cd.predict_values(xe)
        # plt.plot(xe, cl_eval)
        # plt.show()

        # plt.plot(xe, cd_eval)
        # plt.show()




    def define(self):
        n = self.parameters['num_nodes']

        # inputs:
        self.add_input('alpha', shape=n)
        #self.add_input('mach', shape=n)

        # outputs: cl, cd
        self.add_output('cl', shape=n)
        self.add_output('cd', shape=n)

        self.declare_derivatives('cl', 'alpha')
        #self.declare_derivatives('cl', 'mach')
        self.declare_derivatives('cd', 'alpha')
        #self.declare_derivatives('cd', 'mach')

    def compute(self, inputs, outputs):
        n = self.parameters['num_nodes']

        point = np.zeros([n, 1])
        point[:,0] = inputs['alpha']
        cl = self.sm_cl.predict_values(point)
        cd = self.sm_cd.predict_values(point)

        outputs['cl'] = 1*cl
        outputs['cd'] = 1*cd

    def compute_derivatives(self, inputs, derivatives):
        n = self.parameters['num_nodes']

        point = np.zeros([n, 1])
        point[:,0] = inputs['alpha']

        dcl_da = self.sm_cl.predict_derivatives(point, 0)
        #dcl_dm = self.sm_cl.predict_derivatives(point, 1)
        dcd_da = self.sm_cd.predict_derivatives(point, 0)
        #dcd_dm = self.sm_cd.predict_derivatives(point, 1)

        derivatives['cl', 'alpha'] = np.diag(dcl_da.flatten())
        #derivatives['cl', 'mach'] = np.diag(dcl_dm.flatten())
        derivatives['cd', 'alpha'] = np.diag(dcd_da.flatten())
        #derivatives['cd', 'mach'] = np.diag(dcd_dm.flatten())








if __name__ == '__main__':
    
    sim = python_csdl_backend.Simulator(Aero(num_nodes=2, wing_area=19.6))
    sim.run()

    print('C_L: ', sim['C_L'])
    print('C_D: ', sim['C_D'])

    sim.check_partials(step=1E-6, compact_print=True)