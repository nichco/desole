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


uxu = np.array([1478.60216861,1498.31470337,1515.70669826,1534.32788225,1554.93035665,
 1577.20977083,1592.47574373,1603.27247772,1611.10551639,1451.41269526,
 1166.83243067,1082.32234382,1011.41409786, 968.14123336, 934.16988391,
  908.49280534, 887.75305781, 870.55302742, 856.79617618, 845.33712342,
  835.54381438, 827.03265171, 819.85871236, 813.40770417, 807.65611136,
  802.90376728, 798.43370765, 794.47961623, 791.03986669, 787.99380008,
  785.27969353, 782.65401116, 780.41556355, 778.34655732, 776.50900531,
  774.75744928, 773.29783687, 771.97005393, 770.99516333, 769.71474992])
uzu = np.array([1.19914498e+03,1.20314416e+03,1.19522943e+03,1.18535395e+03,
 1.17380671e+03,1.15710198e+03,7.83996493e+02,6.54238417e+02,
 4.48560261e+02,2.08129749e+02,4.35765726e-12,6.08813276e-04,
 6.90743780e-03,3.92150890e-02,1.17052983e-06,4.11456658e-03,
 2.92720776e-03,4.04768543e-03,9.72088104e-03,3.13699615e-03,
 1.19664319e-03,1.72548313e-04,4.52949676e-04,9.82175912e-04,
 4.08037759e-03,7.88449886e-04,1.11280376e-03,3.09799611e-03,
 6.43624782e-04,4.94105147e-04,4.72993706e-04,3.61063450e-04,
 8.34396556e-05,1.10791577e-04,4.69237046e-04,7.66395555e-04,
 1.55558068e-03,6.24237645e-04,1.80709496e-03,5.07879455e-03])
uau = np.array([-0.01350389,-0.03104296,-0.02994083,-0.03743226,-0.04973968,-0.03845421,
  0.14958518, 0.15123054, 0.16194037, 0.15171894, 0.13355846, 0.11829502,
  0.10777767, 0.09956571, 0.09286108, 0.08716222, 0.08222364, 0.07782967,
  0.07388016, 0.07033121, 0.06698612, 0.06397175, 0.06110693, 0.05842687,
  0.05596889, 0.05354298, 0.05130862, 0.04920144, 0.04709851, 0.04529791,
  0.04323155, 0.04161012, 0.03976235, 0.03811269, 0.03656443, 0.03495403,
  0.03365892, 0.03154088, 0.03183971, 0.02871176])

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        dt = self.create_input('dt', options['dt'])
        h_vec = csdl.expand(dt, num-1)
        self.register_output('hvec', h_vec)
        
        # add dynamic inputs to the csdl model
        # ux = self.create_input('ux', val=np.ones((num))*1000) # 1000
        # uz = self.create_input('uz', val=np.ones((num))*1000) # 1000
        # ua = self.create_input('ua', val=np.ones((num))*0.1) # pitch angle (theta)
        
        ux = self.create_input('ux', val=uxu)
        uz = self.create_input('uz', val=uzu)
        ua = self.create_input('ua', val=uau)
        

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
options['dt'] = 1.8505198
options['mass'] = 3000 # (kg)
options['wing_area'] = 19.6 # (m^2)
options['lift_rotor_diameter'] = 2.4 # (m)
options['cruise_rotor_diameter'] = 2.6 # (m)



num = 40
ODEProblem = ODEProblemTest('GaussLegendre4', 'time-marching', num_times=num, display='default', visualization='end')
sim = python_csdl_backend.Simulator(Run(options=options), analytics=True)
sim.run()
#plt.show()
#sim.check_partials(compact_print=False)
#sim.check_totals(step=1E-6)


# prob = CSDLProblem(problem_name='Trajectory Optimization', simulator=sim)
# optimizer = SLSQP(prob, maxiter=5000, ftol=1E-6)
# optimizer.solve()
# optimizer.print_results()


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






print(sim['dvx'])