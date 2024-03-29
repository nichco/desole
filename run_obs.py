import csdl
import numpy as np
import matplotlib.pyplot as plt
import python_csdl_backend
from odeproblemtest3 import ODEProblemTest
from obs import Obs
from modopt.scipy_library import SLSQP
#from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)




uxu = np.array([1474.97030304,1486.00384281,1501.66352753,1517.59669364,1527.54644802,
 1530.3622693 ,1528.38687639,1526.48487116,1527.33371116,1529.57056401,
 1534.75489465,1545.85255722,1569.29460672,1592.53492582,1606.6259998 ,
 1621.3836575 ,1525.82192282,1298.81275957,1278.68352611,1254.05689477,
 1256.07124387,1263.80870652,1277.0583167 ,1293.01370607,1312.93015471,
 1329.64590267,1340.59645955,1347.81832722,1351.19967838,1352.63518067,
 1354.05265324,1355.17937215,1356.3093673 ,1357.09671196,1356.85847974,
 1352.80207319,1341.06459977,1317.49331868,1279.79374891,1267.43039807])
uzu = np.array([1.20090125e+03,1.19961379e+03,1.19770202e+03,1.19875451e+03,
 1.20011941e+03,1.20024333e+03,1.19790868e+03,1.19441924e+03,
 1.19426438e+03,1.19373525e+03,1.19325088e+03,1.19007092e+03,
 1.18566793e+03,8.08923089e+02,6.28893340e+02,4.34082519e+02,
 1.04826939e+02,0.00000000e+00,8.48969910e-04,6.17940451e-12,
 9.96672588e-05,1.91203786e-05,1.78630372e-06,6.59929720e-05,
 7.09993990e-05,7.00975440e-05,1.32088186e-04,1.85646323e-04,
 2.45359940e-04,2.78924995e-04,2.75385148e-04,2.73558469e-04,
 2.69707910e-04,2.34105693e-04,1.82157527e-04,1.35832644e-04,
 8.65769889e-05,1.52930182e-04,7.00006062e-04,1.47208148e-03])
uau = np.array([ 0.12492133, 0.02952486,-0.01895811, 0.06973965, 0.18792869, 0.30309727,
  0.34906585, 0.29414646, 0.28210215, 0.2417602 , 0.18318552, 0.02383716,
 -0.17371992, 0.15069567, 0.15129883, 0.16074984, 0.14558076, 0.11546069,
  0.09604475, 0.08014864, 0.06720266, 0.05440823, 0.04697408, 0.02844422,
  0.04217141, 0.07350207, 0.10041125, 0.1212363 , 0.1294627 , 0.13060295,
  0.13129713, 0.1327248 , 0.1352967 , 0.14183717, 0.15871762, 0.18448351,
  0.21329906, 0.23459098, 0.13656447,-0.03929315])



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        dt = self.create_input('dt', options['dt'])
        h_vec = csdl.expand(dt, num-1)
        self.register_output('hvec', h_vec)
        
        # add dynamic inputs to the csdl model
        #ux = self.create_input('ux', val=np.ones((num))*1000)
        #uz = self.create_input('uz', val=np.ones((num))*1000)
        #ua = self.create_input('ua', val=np.ones((num))*0.1) # pitch angle (theta)
        
        ux = self.create_input('ux', val=uxu)
        uz = self.create_input('uz', val=uzu)
        ua = self.create_input('ua', val=uau)

        # initial conditions for states
        self.create_input('vx_0', 0.1)
        self.create_input('vz_0', 0)
        self.create_input('x_0', 0)
        self.create_input('z_0', 0)
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

        self.register_output('min_z', csdl.min(100*z)/100)
        #self.add_constraint('min_z', lower=-0.1, scaler=1E2)

        # final velocity constraint:
        v = self.register_output('v', (vx**2 + vz**2)**0.5)
        self.register_output('final_v', v[-1])
        self.add_constraint('final_v', equals=60, scaler=1E-1)
        
        cruise_power = self.declare_variable('cruise_power',shape=(num,))
        lift_power = self.declare_variable('lift_power',shape=(num,))
        self.register_output('max_cruise_power', csdl.max(0.0001*cruise_power)/0.0001)
        self.register_output('max_lift_power', csdl.max(0.0001*lift_power)/0.0001)
        self.add_constraint('max_cruise_power', upper=468300, scaler=1E-5)
        self.add_constraint('max_lift_power', upper=170000, scaler=1E-5) # 133652

        
        ag = self.register_output('ag', ((dvx**2 + dvz**2)**0.5)/9.81)
        self.register_output('max_g', csdl.max(10*ag)/10)
        #self.add_constraint('max_g', upper=1.0, scaler=1E0)

        self.register_output('final_gamma', gamma[-1])
        self.add_constraint('final_gamma', equals=0,)

        self.register_output('max_x', csdl.max(x))
        self.add_constraint('max_x', upper=3000, scaler=1E-3)


        eps = 1E-1
        self.add(Obs(num_nodes=num), name='Obs')
        obsi = self.declare_variable('obsi', shape=(num))
        obs_res = self.register_output('obs_res', z - (obsi - eps))
        min_obs_res = self.register_output('min_obs_res', csdl.min(10*obs_res)/10)
        self.add_constraint('min_obs_res', lower=0, scaler=1E0)
        # self.add_constraint('obs_res', lower=0, scaler=1E1)
        
        
        # self.print_var(obs_res)
        self.print_var(min_obs_res)


        
        # compute the total energy:
        self.register_output('energy', e[-1])
        self.print_var(e[-1])
        
        
        # for the minimum energy objective:
        self.add_design_variable('ua', lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=6)
        self.add_design_variable('ux', lower=0, upper=4000, scaler=1E-3)
        self.add_design_variable('uz', lower=0, upper=4000, scaler=1E-3)
        self.add_design_variable('dt', lower=1.0, scaler=1E0) #1E0
        self.add_objective('energy', scaler=1E-2)







options = {}
options['dt'] = 1.82811623 # 2
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

#plt.plot(sim['lift'])
#plt.plot(sim['drag'])
#plt.show()

#plt.plot(sim['density'])
#plt.plot(sim['alpha'])
#plt.plot(sim['gamma'])
#plt.show()

#plt.plot(sim['cruise_vaxial'])
#plt.plot(sim['cruise_vtan'])
#plt.show()
#exit()

prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=2000, ftol=1E-5)
optimizer.solve()
optimizer.print_results()


#print('ux: ', sim['ux'])
#print('uz: ', sim['uz'])
#print('ua: ', sim['ua'])
print(sim['dt'])
print(np.array2string(sim['ux'],separator=','))
print(np.array2string(sim['uz'],separator=','))
print(np.array2string(sim['ua'],separator=','))

print(sim['ag'])

plt.show()


plt.plot(sim['ux'])
plt.plot(sim['uz'])
plt.plot(np.rad2deg(sim['ua'])*100)
plt.show()

plt.plot(sim['lift'])
plt.plot(sim['drag'])
plt.show()







