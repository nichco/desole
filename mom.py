import csdl
import numpy as np
import python_csdl_backend
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


class ImplicitOp(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('num_nodes')
        self.parameters.declare('A')

    def define(self):
        name = self.parameters['name']
        n = self.parameters['num_nodes']
        A = self.parameters['A']

        VIN = self.declare_variable(name + '_vaxial', shape=(n))
        V = (VIN**2 + 1E-12)**0.5
        rho = self.declare_variable('density', shape=(n))
        T = self.declare_variable('T', shape=(n))
        P = self.declare_variable(name + '_power', shape=(n))

        #res = 0.5*T*V*(((T/(0.5*rho*V**2*A)) + 1)**(0.5) + 1) - P

        res = T*V + T*((-V/2) + (((V**2)/4) + (T/(2*rho*A)))**0.5) - P # Chauhan and Martins
        self.register_output('res', res)



class Prop(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('num_nodes')
        self.parameters.declare('d')

    def define(self):
        name = self.parameters['name']
        n = self.parameters['num_nodes']
        d = self.parameters['d']
        A = np.pi*(d/2)**2

        vaxial = self.declare_variable(name + '_vaxial', shape=(n), val=30)
        density = self.declare_variable('density', shape=(n), val=1.225)
        power = self.declare_variable(name + '_power', shape=(n), val=10000)

        solve_res = self.create_implicit_operation(ImplicitOp(name=name,A=A,num_nodes=n))
        solve_res.declare_state(state='T', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False,atol=1E-7,)
        solve_res.linear_solver = csdl.ScipyKrylov()
        T = solve_res(vaxial, density, power)

        self.register_output(name + '_thrust', 1*T)





if __name__ == '__main__':

    name = 'test'
    sim = python_csdl_backend.Simulator(Prop(name=name, d=2, num_nodes=2))
    sim.run()