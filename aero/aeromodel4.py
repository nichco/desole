import numpy as np
import csdl
import python_csdl_backend
from smt.surrogate_models import RBF
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)




xt_cl = np.deg2rad(np.array([-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,]))
yt_cl = np.array([-0.01,-0.35,-0.65,-0.85,-1,-1.12,-1.18,-1.22,-1.235,-1.24,-1.22,-1.19,-1.2,-1.29,-1.28,-1.07869,-0.67686,-0.27838,0.12907,0.5325,0.92638,1.28286,1.35463,
1.34,1.23,1.18,1.19,1.24,1.28,1.29,1.28,1.265,1.22,1.14,1.02,0.85,0.64,0.35,0.01,])

xt_cd = np.deg2rad(np.linspace(-90, 90, 37))
yt_cd = np.array([1.191308205,1.188,1.186033965,1.179762265,1.158815985,1.113876301,1.047947196,0.965115078,0.866730629,0.750070161,0.620489728,0.481112557,
0.335140556,0.186836785,0.067410268,0.028389972,0.006013008,0.000950904,0.013505923,0.043538331,0.116663427,0.266083605,0.420373542,0.57145064,0.721261971,
0.861188551,0.985827271,1.094205044,1.177593611,1.240456117,1.282483901,1.296,1.299,1.302,1.305,1.308,1.311,])

sm_cl = RBF(d0=0.3,print_global=False,print_solver=False,)
sm_cl.set_training_values(xt_cl, yt_cl)
sm_cl.train()

sm_cd = RBF(d0=0.35,print_global=False,print_solver=False,)
sm_cd.set_training_values(xt_cd, yt_cd)
sm_cd.train()



class Aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('wing_area')

    def define(self):
        num = self.parameters['num_nodes']
        s = self.parameters['wing_area']

        alpha = self.declare_variable('alpha', shape=(num))
        v = self.declare_variable('v', shape=(num))
        rho = self.declare_variable('density', shape=(num), val=1.225)

        # custom operation insertion
        cl, cd = csdl.custom(alpha, op=AeroExplicit(num_nodes=num))
        C_L = self.register_output('C_L', 1*cl)
        C_D = self.register_output('C_D', 1*cd)

        lift = 0.5*rho*(v**2)*s*C_L
        drag = 0.5*rho*(v**2)*s*C_D
        self.register_output('lift', lift)
        self.register_output('drag', drag)

class AeroExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        # input: alpha
        self.add_input('alpha', shape=n)

        # output: pressure and density
        self.add_output('cl', shape=n)
        self.add_output('cd', shape=n)

        self.declare_derivatives('cl', 'alpha')
        self.declare_derivatives('cd', 'alpha')

    def compute(self, inputs, outputs):
        n = self.parameters['num_nodes']

        # surrogate model
        # cl = sm_cl.predict_values(inputs['alpha'])
        # cd = sm_cd.predict_values(inputs['alpha'])
        cl = np.zeros((n))
        cd = np.zeros((n))
        for i in range(n):
            a = np.array([inputs['alpha'][i]])
            cl[i] = sm_cl.predict_values(a)
            cd[i] = sm_cd.predict_values(a)

        outputs['cl'] = 1*cl
        outputs['cd'] = 1*cd

    def compute_derivatives(self, inputs, derivatives):
        n = self.parameters['num_nodes']

        # dcl_dalpha = sm_cl.predict_derivatives(inputs['alpha_w'], 0)
        # dcd_dalpha = sm_cd.predict_derivatives(inputs['alpha_w'], 0)
        dcl_dalpha = np.zeros((n))
        dcd_dalpha = np.zeros((n))
        for i in range(n):
            a = np.array([inputs['alpha'][i]])
            dcl_dalpha[i] = sm_cl.predict_derivatives(a, 0)
            dcd_dalpha[i] = sm_cd.predict_derivatives(a, 0)

        derivatives['cl', 'alpha'] = np.diag(dcl_dalpha)
        derivatives['cd', 'alpha'] = np.diag(dcd_dalpha)



if __name__ == '__main__':
    
    sim = python_csdl_backend.Simulator(Aero(num_nodes=2, wing_area=19.6))
    sim.run()

    print('C_L: ', sim['C_L'])
    print('C_D: ', sim['C_D'])

    sim.check_partials(step=1E-6, compact_print=True)