import numpy as np 
import csdl
import python_csdl_backend
from lsdo_rotor.core.pitt_peters.pitt_peters_model import PittPetersModel
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


num_nodes = 1
num_radial = 30
num_tangential = num_azimuthal = 30
thrust_vector =  np.array([[1,0,0]])
thrust_origin =  np.array([[0,0,0]])
reference_point = np.array([0,0,0])
shape = (num_nodes,num_radial,num_tangential)

radius = 1.2
chord = np.linspace(0.3,0.14,num_radial)
twist = np.linspace(50,10,num_radial)*np.pi/180


class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('u')
        self.parameters.declare('v')
        self.parameters.declare('rpm')
    def define(self):
        u = self.parameters['u']
        v = self.parameters['v']
        rpm = self.parameters['rpm']

        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=radius)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=chord)
        self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=twist)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm', val=rpm)
        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=u)
        self.create_input(name='v', shape=(num_nodes, 1), units='m/s', val=v)
        self.create_input(name='w', shape=(num_nodes, 1), units='m/s', val=0)
        self.create_input(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='r', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Theta', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Psi', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='x', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='y', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=1000)
                
        self.add(PittPetersModel(   
            name='propulsion',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_azimuthal,
            airfoil='NACA_4412',
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=3,
        ),name='pitt_peters_model')



"""
sim = python_csdl_backend.Simulator(Run(u=0,v=0,rpm=1500))
sim.run()

print(sim['C_T'].flatten())
print(sim['C_P'].flatten())
"""



# scripting:
n = np.linspace(500,5000,10) # rotor speed (rpm)
vaxial = np.linspace(-30,100,10) # axial inflow (m/s)
vtan = np.linspace(0,100,10) # edgewise inflow (m/s)
datact = np.zeros((len(n), len(vaxial), len(vtan)))
datacp = np.zeros((len(n), len(vaxial), len(vtan)))

for k, rpm in enumerate(n):
    for i, u in enumerate(vaxial):
        for j, v in enumerate(vtan):
            sim = python_csdl_backend.Simulator(Run(u=u,v=v,rpm=rpm))
            sim.run()
            datact[k,i,j] = sim['C_T'].flatten()
            datacp[k,i,j] = sim['C_P'].flatten()


print(datact)


plt.contourf(vaxial,vtan,datact[0,:,:])
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datact[1,:,:])
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datact[2,:,:])
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datact[3,:,:])
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datact[4,:,:])
plt.colorbar(shrink=1)
plt.show()

plt.contourf(vaxial,vtan,datact[5,:,:])
plt.colorbar(shrink=1)
plt.show()

file = open('ct2.pkl', 'wb')
pickle.dump(datact, file)

file = open('cp2.pkl', 'wb')
pickle.dump(datacp, file)