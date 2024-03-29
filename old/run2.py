import csdl
import numpy as np
import matplotlib.pyplot as plt
import python_csdl_backend
from odeproblemtest2 import ODEProblemTest
from modopt.scipy_library import SLSQP
#from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
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
        ux = self.create_input('ux', val=np.ones((num))*2000)
        uz = self.create_input('uz', val=np.ones((num))*1000)
        ua = self.create_input('ua', val=np.ones((num))*0) # pitch angle (theta)

        # initial conditions for states
        self.create_input('u_0', 0.05)
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
        du = self.declare_variable('du', shape=(num,))
        dw = self.declare_variable('dw', shape=(num,))

        # final altitude constraint:
        self.register_output('final_z', z[-1])
        self.add_constraint('final_z', equals=300, scaler=1E-2)

        self.register_output('min_z', csdl.min(1000*z)/1000)
        #self.add_constraint('z', lower=-0.1)
        self.add_constraint('min_z', lower=-0.1)

        # final velocity constraint:
        v = (u**2 + w**2)**0.5
        self.register_output('final_v', v[-1])
        self.add_constraint('final_v', equals=50, scaler=1E-1)
        
        cruise_power = self.declare_variable('cruise_power',shape=(num,))
        lift_power = self.declare_variable('lift_power',shape=(num,))
        self.register_output('max_cruise_power', csdl.max(cruise_power))
        self.register_output('max_lift_power', csdl.max(lift_power))

        self.add_constraint('max_cruise_power', upper=468300, scaler=1E-5)
        self.add_constraint('max_lift_power', upper=133652, scaler=1E-5)
        
        alpha = self.declare_variable('alpha', shape=(num,))
        gamma = self.register_output('gamma', ua - alpha)
        self.register_output('final_gamma', gamma[-1])
        self.add_constraint('final_gamma', equals=0)
        
        #self.register_output('theta_0', ua[0])
        #self.add_constraint('theta_0', equals=0)

        #dtheta = self.create_output('dtheta',shape=(num-1,), val=0)
        #for i in range(1,num):
        #    dtheta[i-1] = (((ua[i] - ua[i-1])/dt)**2)**0.5

        #a = self.register_output('a', ((du**2 + dw**2)**0.5)/9.81)
        #self.add_constraint('a', upper=0.5)

 
        
        # compute the total energy:
        self.register_output('energy', e[-1])
        self.print_var(e[-1])
        
        
        # for the minimum energy objective:
        self.add_design_variable('ua', lower=np.deg2rad(-20),upper=np.deg2rad(20),scaler=6)
        self.add_design_variable('ux', lower=0, upper=5000, scaler=1E-3)
        self.add_design_variable('uz', lower=0, upper=5000, scaler=1E-3)
        self.add_design_variable('dt', lower=0.5, scaler=1E0)
        self.add_objective('energy', scaler=1E-2)







options = {}
options['dt'] = 2
options['mass'] = 3000 # (kg)
options['wing_area'] = 19.6 # (m^2)
options['lift_rotor_diameter'] = 2.4 # (m)
options['cruise_rotor_diameter'] = 2.6 # (m)



num = 30
ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times=num, display='default', visualization='end')
sim = python_csdl_backend.Simulator(Run(options=options), analytics=0)
#sim.run()

#sim.check_partials(compact_print=False)
#sim.check_totals(step=1E-6)


prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=1000, ftol=1E-5)
optimizer.solve()
optimizer.print_results()


#print('ux: ', sim['ux'])
#print('uz: ', sim['uz'])
#print('ua: ', sim['ua'])
print(sim['dt'])
print(np.array2string(sim['ux'],separator=','))
print(np.array2string(sim['uz'],separator=','))
print(np.array2string(sim['ua'],separator=','))

plt.show()


plt.plot(sim['ux'])
plt.plot(sim['uz'])
plt.plot(np.rad2deg(sim['ua'])*100)
plt.show()

plt.plot(sim['lift'])
plt.plot(sim['drag'])
plt.show()







