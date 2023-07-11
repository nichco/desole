import csdl
import numpy as np
import python_csdl_backend
# from prop.propmodel5 import Prop
from aero.atm import Atm
from aero.aeromodel3 import Aero
# from motor.motormodel import Motor
from mom import Prop



class ODESystemModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')

    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']
        

        # states:
        vx = self.create_input('vx', shape=n)
        vz = self.create_input('vz', shape=n, val=0.01)
        x = self.create_input('x', shape=n, val=0)
        z = self.create_input('z', shape=n, val=0)
        e = self.create_input('e', shape=n, val=0)

        # parameters:
        ux = self.declare_variable('ux', shape=(n))
        uz = self.declare_variable('uz', shape=(n))
        ua = self.declare_variable('ua', shape=(n))

        # options:
        m = options['mass']
        g = 9.81

        # compute velocity and alpha:
        v = self.register_output('v', (vx**2 + vz**2)**0.5)
        #self.print_var(v)
        gamma = self.register_output('gamma', csdl.arctan(vz/vx))
        alpha = self.register_output('alpha', ua - gamma)

        # the atmosphere model:
        self.add(Atm(num_nodes=n), name='Atm')
        
        # the aerodynamic model:
        self.add(Aero(num_nodes=n, wing_area=options['wing_area']), name='Aero2')
        L = self.declare_variable('lift', shape=(n))
        D = self.declare_variable('drag', shape=(n))


        # rotor models
        self.register_output('cruise_vaxial', v*csdl.cos(alpha))
        self.register_output('cruise_power', 1*ux)
        self.add(Prop(name='cruise',d=options['cruise_rotor_diameter'],num_nodes=n), name='CruiseProp',
                 promotes=['cruise_thrust', 'cruise_power', 'density', 'cruise_vaxial'])
        tc = self.declare_variable('cruise_thrust', shape=(n))
        cruise_power = 1*ux


        self.register_output('lift_vaxial', v*csdl.sin(alpha))
        self.register_output('lift_power', 1*uz)
        self.add(Prop(name='lift', d=options['lift_rotor_diameter'],num_nodes=n), name='LiftProp', 
                 promotes=['lift_thrust', 'lift_power', 'lift_vaxial', 'density'])
        tl = 8*self.declare_variable('lift_thrust', shape=(n))
        lift_power = 8*uz

        
        # system of ODE's
        dvx = (tc*csdl.cos(ua) - tl*csdl.sin(ua) - D*csdl.cos(gamma) - L*csdl.sin(gamma))/m
        dvz = (tc*csdl.sin(ua) + tl*csdl.cos(ua) - D*csdl.sin(gamma) + L*csdl.cos(gamma) - m*g)/m
        dx = 1*vx
        dz = 1*vz


        de = 1E-4*(cruise_power + lift_power)

        # register outputs
        self.register_output('dvx', dvx)
        self.register_output('dvz', dvz)
        self.register_output('dx', dx)
        self.register_output('dz', dz)
        self.register_output('de', de)

 




       
if __name__ == '__main__':
    options = {}
    options['dt'] = 3
    options['mass'] = 3000 # (kg)
    options['wing_area'] = 19.6 # (m^2)
    options['lift_rotor_diameter'] = 2.4 # (m)
    options['cruise_rotor_diameter'] = 2.6 # (m)
    
    sim = python_csdl_backend.Simulator(ODESystemModel(num_nodes=20, options=options))
    sim.run()

    sim.check_partials(step=1E-6, compact_print=True)
    

    
        
        
        
        