import csdl
import numpy as np
import matplotlib.pyplot as plt
import python_csdl_backend
from odeproblemtest2 import ODEProblemTest
from modopt.scipy_library import SLSQP
#from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
#import matplotlib as mpl
plt.rcParams.update(plt.rcParamsDefault)





class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        dt = self.create_input('dt', options['dt'])
        h_vec = csdl.expand(dt, num-1)
        self.register_output('hvec', h_vec)
        
        # add dynamic inputs to the csdl model
        self.create_input('ux', val=np.ones((num))*2500)
        self.create_input('uz', val=np.ones((num))*1000)
        self.create_input('ua', val=np.ones((num))*0) # pitch angle (theta)

        # initial conditions for states
        self.create_input('u_0', 0.1)
        self.create_input('w_0', 0)
        self.create_input('x_0', 0)
        self.create_input('z_0', 0)
        self.create_input('e_0', 0)

        # create model containing the integrator
        optionsdict = {'options': options}
        self.add(ODEProblem.create_solver_model(ODE_parameters=optionsdict, profile_parameters=optionsdict), 'subgroup')

        # declare variables from integrator
        u = self.declare_variable('u', shape=(num,))
        w = self.declare_variable('w', shape=(num,))
        x = self.declare_variable('x', shape=(num,))
        z = self.declare_variable('z', shape=(num,))
        e = self.declare_variable('e', shape=(num,))

        # final altitude constraint:
        self.register_output('final_z', z[-1])
        self.add_constraint('final_z', equals=300, scaler=1E-2)
        self.add_constraint('z', lower=-1)

        self.add_constraint('x', lower=-1)

        # final velocity constraint:
        self.register_output('final_u', u[-1])
        self.add_constraint('final_u', equals=50, scaler=1E-1)
 
        
        # compute the total energy:
        self.register_output('energy', e[-1])
        self.print_var(e[-1])
       
        
        
        # for the minimum energy objective:
        self.add_design_variable('ua',lower=-np.pi/4,upper=np.pi/4,scaler=1E1)
        self.add_design_variable('ux',lower=0, upper=5000, scaler=1E-4)
        self.add_design_variable('uz',lower=0, upper=5000, scaler=1E-4)
        #self.add_design_variable('dt',lower=2.0, scaler=1E0)
        self.add_objective('energy', scaler=1E0)







options = {}
options['dt'] = 4
options['mass'] = 3000 # (kg)
options['wing_area'] = 19.6 # (m^2)
options['lift_rotor_diameter'] = 2.4 # (m)
options['cruise_rotor_diameter'] = 2.6 # (m)

num = 15
ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times=num, display='default', visualization='end')
sim = python_csdl_backend.Simulator(Run(options=options), analytics=0)
#im.run()

#sim.check_partials(compact_print=False)
#sim.check_totals(step=1E-6)


prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=1000, ftol=1E-4)
optimizer.solve()
optimizer.print_results()


print('ux: ', sim['ux'])
print('uz: ', sim['uz'])
print('ua: ', sim['ua'])

plt.show()

#print(sim['v'])
#print(sim['x'])
#print(sim['gamma'])
#print(sim['h'])
#print(sim['e'])