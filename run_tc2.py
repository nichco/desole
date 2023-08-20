import csdl
import numpy as np
import matplotlib.pyplot as plt
import python_csdl_backend
from odeproblemtest3 import ODEProblemTest
from modopt.scipy_library import SLSQP
#from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


"""
uxu = np.array([1481.09221058,1504.35719711,1522.76230455,1536.76925957,1430.91669911,
 1268.42905167,1130.8019644 ,1038.67825902,1030.81354477,1091.44467212,
 1159.43408372,1210.92889466,1245.91380548,1268.98454785,1283.96454692,
 1297.03860693,1308.54894824,1318.83577428,1324.09019672,1325.50739786,
 1325.52649957,1325.03176855,1324.41776828,1323.8818692 ,1323.48009187,
 1323.18877549,1322.94336676,1322.64861961,1322.18665795,1321.35645065,
 1319.80663981,1317.24160898,1312.76273997,1304.88397405,1294.38449178,
 1279.2658348 ,1258.86624623,1235.26792909,1218.71114783,1212.70963334])
uzu = np.array([1.12115860e+03,1.05624724e+03,9.98565853e+02,8.93518949e+02,
 7.53475398e+02,5.88568364e+02,4.07047174e+02,1.96043982e+02,
 9.98477824e-16,4.61264806e-15,3.29779015e-03,1.47685442e-05,
 0.00000000e+00,2.62783150e-04,6.37129949e-04,1.32077916e-16,
 3.23000456e-14,1.23858001e-04,5.19140592e-04,1.90548120e-03,
 1.71650407e-04,4.81355382e-03,6.43034629e-15,1.75768083e-03,
 1.05815871e-03,5.15937942e-04,1.19513046e-03,3.11105944e-04,
 4.72787474e-04,1.72436145e-04,4.82190118e-04,1.13097464e-04,
 9.38847772e-04,6.21197058e-04,4.15296145e-15,1.26790654e-03,
 6.92008224e-06,9.44056072e-04,1.93737570e-14,2.05504828e-02])
uau = np.array([-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,
 -0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.32606395,-0.25300459,
 -0.17691378,-0.094814  ,-0.01541509, 0.05668532, 0.11682219, 0.16083719,
  0.16670529, 0.15702758, 0.14860919, 0.1409093 , 0.13485716, 0.13065325,
  0.12821729, 0.12746971, 0.1284496 , 0.13136246, 0.13662026, 0.14484459,
  0.15671018, 0.17244195, 0.19112982, 0.21055698, 0.22732882, 0.23741101,
  0.22067169, 0.14319626, 0.05921549,-0.04880338])
"""

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        dt = self.create_input('dt', options['dt'])
        h_vec = csdl.expand(dt, num-1)
        self.register_output('hvec', h_vec)
        
        # add dynamic inputs to the csdl model
        ux = self.create_input('ux', val=np.ones((num))*1000) # 1000
        uz = self.create_input('uz', val=np.ones((num))*1000) # 1000
        ua = self.create_input('ua', val=np.ones((num))*0.1) # pitch angle (theta)
        
        #ux = self.create_input('ux', val=uxu)
        #uz = self.create_input('uz', val=uzu)
        #ua = self.create_input('ua', val=uau)
        

        # initial conditions for states
        self.create_input('vx_0', 0.1)
        self.create_input('vz_0', 0)
        self.create_input('x_0', 0)
        self.create_input('z_0', 300) # 300
        self.create_input('e_0', 0)

        # create model containing the integrator
        optionsdict = {'options': options}
        self.add(ODEProblem.create_solver_model(ODE_parameters=optionsdict, profile_parameters=optionsdict), 'subgroup')

        # declare variables from integrator
        vx = self.declare_variable('vx', shape=(num,))
        vz = self.declare_variable('vz', shape=(num,))
        x = self.declare_variable('x', shape=(num,))
        z = self.declare_variable('z', shape=(num,))
        e = self.declare_variable('e', shape=(num,))
        alpha = self.declare_variable('alpha', shape=(num,))
        dvx = self.declare_variable('dvx', shape=(num,))
        dvz = self.declare_variable('dvz', shape=(num,))
        gamma = self.declare_variable('gamma', shape=(num,))

        # final altitude constraint:
        self.register_output('final_z', z[-1])
        self.add_constraint('final_z', equals=300, scaler=1E-2)

        #self.register_output('min_z', csdl.min(100*z)/100)
        #self.add_constraint('min_z', lower=-0.1, scaler=1E2)
        min_z = self.register_output('min_z', csdl.min(10*z)/10)
        self.add_constraint('min_z', lower=250, scaler=1E-2)
        self.print_var(min_z)

        # final velocity constraint:
        v = self.register_output('v', (vx**2 + vz**2)**0.5)
        self.register_output('final_v', v[-1])
        self.add_constraint('final_v', equals=60, scaler=1E-1) # 50
        
        cruise_power = self.declare_variable('cruise_power',shape=(num,))
        lift_power = self.declare_variable('lift_power',shape=(num,))
        self.print_var(cruise_power)
        max_cruise_power = self.register_output('max_cruise_power', csdl.max(0.00001*cruise_power)/0.00001)
        max_lift_power = self.register_output('max_lift_power', csdl.max(0.00001*lift_power)/0.00001)
        self.add_constraint('max_cruise_power', upper=468300, scaler=1E-5) # 1E-5
        self.add_constraint('max_lift_power', upper=170000, scaler=1E-5) # 133652
        
        self.print_var(max_cruise_power)
        self.print_var(max_lift_power)
        
        ag = self.register_output('ag', ((dvx**2 + dvz**2)**0.5)/9.81)
        # self.print_var(ag)
        max_g = self.register_output('max_g', csdl.max(10*ag)/10)
        self.print_var(max_g)
        self.add_constraint('max_g', upper=1.0, scaler=1E1)

        self.register_output('final_gamma', gamma[-1])
        self.add_constraint('final_gamma', equals=0,)

        self.register_output('max_x', csdl.max(x))
        #self.add_constraint('max_x', upper=3000, scaler=1E-3)


        
        # compute the total energy:
        self.register_output('energy', e[-1])
        self.print_var(e[-1])
        
        
        # for the minimum energy objective:
        self.add_design_variable('ua', lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=6)
        self.add_design_variable('ux', lower=0, upper=4000, scaler=1E-3)
        self.add_design_variable('uz', lower=0, upper=4000, scaler=1E-3)
        self.add_design_variable('dt', lower=1.0, scaler=1E1) # lower=1.0
        # self.add_design_variable('dt', lower=0.3, scaler=1E0) # for min time
        self.add_objective('energy', scaler=1E-2)
        
        # self.add_objective('dt', scaler=1E0)







options = {}
options['dt'] = 2 # 2
options['mass'] = 3000 # (kg)
options['wing_area'] = 19.6 # (m^2)
options['lift_rotor_diameter'] = 2.4 # (m)
options['cruise_rotor_diameter'] = 2.6 # (m)



num = 40
ODEProblem = ODEProblemTest('GaussLegendre4', 'time-marching', num_times=num, display='default', visualization='end')
sim = python_csdl_backend.Simulator(Run(options=options), analytics=0)
#sim.run()
#plt.show()
#sim.check_partials(compact_print=False)
#sim.check_totals(step=1E-6)


prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=5000, ftol=1E-6)
optimizer.solve()
optimizer.print_results()


print(1E-6*sim['energy']/1E-4)

#print('ux: ', sim['ux'])
#print('uz: ', sim['uz'])
#print('ua: ', sim['ua'])
print(sim['dt'])
print(np.array2string(sim['ux'],separator=','))
print(np.array2string(sim['uz'],separator=','))
print(np.array2string(sim['ua'],separator=','))

print(np.array2string(sim['x'],separator=','))
print(np.array2string(sim['z'],separator=','))
print(np.array2string(sim['v'],separator=','))

print(sim['ag'])

plt.show()


plt.plot(sim['ux'])
plt.plot(sim['uz'])
plt.plot(np.rad2deg(sim['ua'])*100)
plt.show()

plt.plot(sim['lift'])
plt.plot(sim['drag'])
plt.show()


#print(np.array2string(sim['cruise_power'].flatten(),separator=','))
#print(np.array2string(sim['lift_power'].flatten(),separator=','))






