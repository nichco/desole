from ozone.api import ODEProblem
from ode import ODESystemModel



class ODEProblemTest(ODEProblem):
    def setup(self):
        self.add_parameter('ua', dynamic=True, shape=(self.num_times))
        self.add_parameter('ux', dynamic=True, shape=(self.num_times))
        self.add_parameter('uz', dynamic=True, shape=(self.num_times))

        self.add_state('v', 'dv', initial_condition_name='v_0', output='v', interp_guess=[0.625, 58])
        self.add_state('gamma', 'dgamma', initial_condition_name='gamma_0', output='gamma')
        self.add_state('h', 'dh', initial_condition_name='h_0', output='h', interp_guess=[0.1, 300])
        self.add_state('x', 'dx', initial_condition_name='x_0', output='x', interp_guess=[0, 3000])
        self.add_state('e', 'de', initial_condition_name='e_0', output='e', interp_guess=[0, 4000])

        self.add_times(step_vector='hvec')

        self.set_ode_system(ODESystemModel)

        # export variables of interest from ode for troubleshooting
        #self.set_profile_system(ODESystemModel)
        #self.add_profile_output('lift')
        #self.add_profile_output('drag')
        #self.add_profile_output('cruisepower')
        #self.add_profile_output('liftpower')
        #self.add_profile_output('dv')
        #self.add_profile_output('dgamma')
