import csdl
import numpy as np
import matplotlib.pyplot as plt
import python_csdl_backend
from odeproblemtest import ODEProblemTest
from modopt.scipy_library import SLSQP
#from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem





class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        dt = self.create_input('dt', options['dt'])
        h_vec = csdl.expand(dt, num-1)
        self.register_output('hvec', h_vec)
        
        # add dynamic inputs to the csdl model
        self.create_input('ux', val=np.ones((num))*1000)
        self.create_input('uz', val=np.ones((num))*1000)
        self.create_input('ua', val=np.ones((num))*0)

        # initial conditions for states
        self.create_input('v_0', 0.1)
        self.create_input('gamma_0', 0)
        self.create_input('h_0', 0)
        self.create_input('x_0', 0)
        self.create_input('e_0', 0)

        # create model containing the integrator
        optionsdict = {'options': options}
        self.add(ODEProblem.create_solver_model(ODE_parameters=optionsdict, profile_parameters=optionsdict), 'subgroup')

        # declare variables from integrator
        v = self.declare_variable('v', shape=(num,))
        x = self.declare_variable('x', shape=(num,))
        gamma = self.declare_variable('gamma', shape=(num,))
        h = self.declare_variable('h', shape=(num,))
        e = self.declare_variable('e', shape=(num,))

        # final altitude constraint:
        self.register_output('final_h', h[-1])
        self.add_constraint('final_h', equals=300, scaler=1E-2)
 
        
        # compute total energy
        self.register_output('energy', e[-1])
        self.print_var(e[-1])
       
        
        
        # for the minimum energy objective
        self.add_design_variable('ua',lower=-np.pi/2,upper=np.pi/2,scaler=4)
        self.add_design_variable('ux',lower=0, scaler=1E-3)
        self.add_design_variable('uz',lower=0, scaler=1E-3)
        self.add_design_variable('dt',lower=2.0,scaler=1E-1)
        self.add_objective('energy', scaler=1E-4)







options = {}
options['dt'] = 3
options['m'] = 3000 # (kg)
options['wing_area'] = 19.6 # (m^2)
options['lift_rotor_diameter'] = 2.4 # (m)
options['cruise_rotor_diameter'] = 2.6 # (m)

num = 20
ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times=num, display='default', visualization='end')
sim = python_csdl_backend.Simulator(Run(options=options), analytics=0)
sim.run()

#sim.check_partials(compact_print=False)
#sim.check_totals(step=1E-6)

"""
prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=1000, ftol=1E-4)
#optimizer = SNOPT(prob,Major_iterations=1000,
#                    Major_optimality=1e-7,
#                    Major_feasibility=1E-7,
#                    append2file=True,
#                    Linesearch_tolerance=0.99,
#                    #Hessian_frequency=10,
#                    Major_step_limit=0.1
#                    )
optimizer.solve()
optimizer.print_results()
# plot states from integrator
plt.show()
"""
