import csdl
import python_csdl_backend
import numpy as np


# standard atmosphere model valid through 11km:
class Atm(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']
        
        h = self.declare_variable('h', shape=n)
        v = self.declare_variable('v', shape=n)
        
        g = 9.806 # m/(s^2)
        a = -6.5E-3 # K/m
        Ts = 288.16 # deg K @ sea level
        Ps = 1.01325E5 # Pascals at sea level
        rhoS = 1.225 # kg/m^3 at sea level
        R = 287 # J/(Kg-K) gas constant
        
        temperature = Ts + a*h
        pressure = Ps*((temperature/Ts)**((-g)/(a*R)))
        density = rhoS*((temperature/Ts)**(-((g/(a*R)) + 1)))

        gamma = 1.4
        a = (gamma*R*temperature)**0.5

        self.register_output('mach', v/a)
        self.register_output('pressure', pressure)
        self.register_output('density', density)




if __name__ == '__main__':
    sim = python_csdl_backend.Simulator(Atm(num_nodes=10))
    sim['h'] = 1000
    sim.run()

    print(sim['pressure'])
    print(sim['density'])

    sim.check_partials(compact_print=True)