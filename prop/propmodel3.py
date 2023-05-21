import csdl
import python_csdl_backend
import numpy as np



class Prop(csdl.Model):
    def initialize(self):
        self.parameters.declare('name',types=str)
        self.parameters.declare('num_nodes')
        self.parameters.declare('d')

    def define(self):
        num = self.parameters['num_nodes']
        name = self.parameters['name']
        d = self.parameters['d']

        rpm_in = self.declare_variable(name + '_rpm', shape=num, val=1500)
        vaxial_in = self.declare_variable(name + '_vaxial', shape=num, val=10)
        vtan_in = self.declare_variable(name + '_vtan', shape=num, val=10)

        rpm = self.register_output('rpm', 1*rpm_in)
        vAxial = self.register_output('vAxial', 1*vaxial_in)
        vTan = self.register_output('vTan', 1*vtan_in)

        C_T = 0.25
        C_P = 0.25

        rho = self.declare_variable('density', shape=(num), val=1.225)
        n = rpm/60
        self.register_output(name + '_thrust', C_T*rho*(n**2)*(d**4))
        self.register_output(name + '_power', C_P*rho*(n**3)*(d**5))