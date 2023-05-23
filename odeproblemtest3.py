from ozone.api import ODEProblem
from ode3 import ODESystemModel



class ODEProblemTest(ODEProblem):
    def setup(self):
        self.add_parameter('ua', dynamic=True, shape=(self.num_times))
        self.add_parameter('ux', dynamic=True, shape=(self.num_times))
        self.add_parameter('uz', dynamic=True, shape=(self.num_times))

        self.add_state('vx', 'dvx', initial_condition_name='vx_0', output='vx', interp_guess=[0.625, 58])
        self.add_state('vz', 'dvz', initial_condition_name='vz_0', output='vz', interp_guess=[1, 1])
        self.add_state('x', 'dx', initial_condition_name='x_0', output='x', interp_guess=[0.1, 300])
        self.add_state('z', 'dz', initial_condition_name='z_0', output='z', interp_guess=[0, 3000])
        self.add_state('e', 'de', initial_condition_name='e_0', output='e', interp_guess=[0, 4000])

        self.add_times(step_vector='hvec')

        self.set_ode_system(ODESystemModel)

        # export variables of interest from ode for troubleshooting
        self.set_profile_system(ODESystemModel)
        self.add_profile_output('alpha')
        self.add_profile_output('gamma')
        self.add_profile_output('lift')
        self.add_profile_output('drag')
        self.add_profile_output('density')
        self.add_profile_output('cruise_power')
        self.add_profile_output('lift_power')
        self.add_profile_output('cruise_vaxial')
        self.add_profile_output('cruise_vtan')
        self.add_profile_output('dvx')
        self.add_profile_output('dvz')
        self.add_profile_output('gamma')
