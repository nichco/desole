import csdl
import numpy as np
import python_csdl_backend
from prop.propmodel import Prop



class ODESystemModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')

    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']
        

        # states:
        v = self.create_input('v', shape=n)
        gamma = self.create_input('gamma', shape=n)
        h = self.create_input('h', shape=n)
        x = self.create_input('x', shape=n)
        e = self.create_input('e', shape=n)

        # parameters:
        ux = self.declare_variable('ux', shape=(n))
        uz = self.declare_variable('uz', shape=(n))
        ua = self.declare_variable('ua', shape=(n))

        # options:
        m = options['mass']
        g = options['gravity']
        
        # the aerodynamic model:
        self.add(aero(num_nodes=n, options=options), name='aero')
        L = self.declare_variable('lift', shape=(n))
        D = self.declare_variable('drag', shape=(n))


        # rotor and motor models
        self.register_output('cruise_vaxial', ((v*csdl.cos(ua))**2)**0.5)
        self.register_output('cruise_vtan', ((v*csdl.sin(ua))**2)**0.5)
        self.register_output('cruise_rpm', 1*ux)
        self.add(Prop(name='cruise', num_nodes=n, d=options['cruise_rotor_diameter']), name='CruiseProp', 
                 promotes=['cruise_thrust', 'cruise_power', 'cruise_rpm', 'cruise_vaxial', 'cruise_vtan', 'density'])
        tc = cruise_thrust = self.declare_variable('cruise_thrust', shape=(n))
        cruise_power = self.declare_variable('cruise_power', shape=(n))

        self.register_output('lift_vaxial', v*csdl.sin(ua))
        self.register_output('lift_vtan', v*csdl.cos(ua))
        self.register_output('lift_rpm', 1*uz)
        self.add(Prop(name='lift', num_nodes=n, d=options['lift_rotor_diameter']), name='LiftProp', 
                 promotes=['lift_thrust', 'lift_power', 'lift_rpm', 'lift_vaxial', 'lift_vtan', 'density'])
        tl = lift_thrust = 8*self.declare_variable('lift_thrust', shape=(n))
        lift_power = 8*self.declare_variable('lift_power', shape=(n))

        
        # system of ODE's
        dv = (tc/m)*csdl.cos(ua) - (tl/m)*csdl.sin(ua) - (D/m) - g*csdl.sin(gamma)
        dgamma = (tc/(m*v))*csdl.sin(ua) + (tl/(m*v))*csdl.cos(ua) + (L/(m*v)) - (g*csdl.cos(gamma)/v)
        dh = v*csdl.sin(gamma)
        dx = v*csdl.cos(gamma)

        cruise_eta = 1
        lift_eta = 1
        de = 1E-8*((cruise_power/cruise_eta) + (lift_power/lift_eta))

        # register outputs
        self.register_output('dv', dv)
        self.register_output('dgamma', dgamma)
        self.register_output('dh', dh)
        self.register_output('dx', dx)
        self.register_output('de', de)

 




       
if __name__ == '__main__':
    pass    
        
        
        
        
        