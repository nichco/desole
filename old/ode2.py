import csdl
import numpy as np
import python_csdl_backend
from prop.propmodel import Prop
from aero.atm import Atm
from aero.aeromodel import Aero



class ODESystemModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')

    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']
        

        # states:
        u = self.create_input('u', shape=n)
        w = self.create_input('w', shape=n, val=0.01)
        x = self.create_input('x', shape=n, val=100)
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
        v = self.register_output('v', (u**2 + w**2)**0.5)
        alpha = self.register_output('alpha', csdl.arctan(w/u))

        # the atmosphere model:
        self.register_output('h', 1*z)
        self.add(Atm(num_nodes=n), name='Atm')
        
        # the aerodynamic model:
        self.add(Aero(num_nodes=n, wing_area=options['wing_area']), name='Aero2')
        L = self.declare_variable('lift', shape=(n))
        D = self.declare_variable('drag', shape=(n))


        # rotor and motor models
        #self.register_output('cruise_vaxial', (u**2 + 1E-14)**0.5)
        self.register_output('cruise_vaxial', 1*u)
        self.register_output('cruise_vtan', (w**2 + 1E-14)**0.5)
        self.register_output('cruise_rpm', 1*ux)
        self.add(Prop(name='cruise', num_nodes=n, d=options['cruise_rotor_diameter']), name='CruiseProp', 
                 promotes=['cruise_thrust', 'cruise_power', 'cruise_rpm', 'cruise_vaxial', 'cruise_vtan', 'density'])
        tc = cruise_thrust = self.declare_variable('cruise_thrust', shape=(n))
        cruise_power = self.declare_variable('cruise_power', shape=(n))


        #self.register_output('lift_vaxial', ((-w)**2 + 1E-14)**0.5)
        self.register_output('lift_vaxial', -1*w)
        self.register_output('lift_vtan', (u**2 + 1E-14)**0.5)
        self.register_output('lift_rpm', 1*uz)
        self.add(Prop(name='lift', num_nodes=n, d=options['lift_rotor_diameter']), name='LiftProp', 
                 promotes=['lift_thrust', 'lift_power', 'lift_rpm', 'lift_vaxial', 'lift_vtan', 'density'])
        tl = lift_thrust = 8*self.declare_variable('lift_thrust', shape=(n))
        lift_power = 8*self.declare_variable('lift_power', shape=(n))

        
        # system of ODE's
        du = (tc - m*g*csdl.sin(ua) + L*csdl.sin(alpha) - D*csdl.cos(alpha))/m
        dw = (-tl + m*g*csdl.cos(ua) - L*csdl.cos(alpha) - D*csdl.sin(alpha))/m
        dx = u*csdl.cos(ua) + w*csdl.sin(ua)
        dz = u*csdl.sin(ua) - w*csdl.cos(ua)

        cruise_eta = 1
        lift_eta = 1
        de = 1E-4*((cruise_power/cruise_eta) + (lift_power/lift_eta))

        # register outputs
        self.register_output('du', du)
        self.register_output('dw', dw)
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
    

    
        
        
        
        