import csdl
import numpy as np
import python_csdl_backend
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


class ImplicitOp(csdl.Model):
    def initialize(self):
        pass

    def define(self):
        V = self.declare_variable('V')
        rho = self.declare_variable('rho')
        A = self.declare_variable('A')
        T = self.declare_variable('T')
        P = self.declare_variable('P')

        res = 0.5*T*V*(((T/(0.5*rho*V**2*A)) + 1)**(0.5) + 1) - P
        self.register_output('res', res)



class Prop(csdl.Model):
    def initialize(self):
        pass

    def define(self):

        V = self.declare_variable('V', val=30)
        rho = self.declare_variable('rho', val=1.225)
        A = self.declare_variable('A', val=3)
        P = self.declare_variable('P', val=10000)

        solve_res = self.create_implicit_operation(ImplicitOp())
        solve_res.declare_state(state='T', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False,atol=1E-7,)
        solve_res.linear_solver = csdl.ScipyKrylov()
        T = solve_res(V, rho, A, P)





if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Prop())
    sim.run()