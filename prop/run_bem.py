import numpy as np 
import csdl
import python_csdl_backend
from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


num_nodes = 1
num_radial = 50
num_tangential = num_azimuthal = 1
thrust_vector =  np.array([[1,0,0]])
thrust_origin =  np.array([[0,0,0]])
reference_point = np.array([0,0,0])
shape = (num_nodes,num_radial,num_tangential)

rotor_radius = 1.2
altitude = 0 # in (m)
num_blades = 3

airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

chord = np.linspace(0.3, 0.2, num_radial)
twist = np.linspace(60, 15, num_radial)

# sim = python_csdl_backend.Simulator(BEMRunModel(
#         rotor_radius=rotor_radius,
#         rpm=4500,
#         Vx=30,
#         altitude=altitude,
#         shape=shape,
#         num_blades=num_blades,
#         airfoil_name='NACA_4412',
#         airfoil_polar=airfoil_polar,
#         chord_distribution=chord,
#         twist_distribution=twist,
#         thrust_vector=thrust_vector,
#         thrust_origin=thrust_origin,))

# sim.run()
# print('C_T: ', sim['C_T'])



vx = np.linspace(0, 75, 15)
rpm = np.linspace(800, 5000, 15)
datact = np.zeros((len(rpm), len(vx)))
datacp = np.zeros((len(rpm), len(vx)))
for i, n in enumerate(rpm):
    for j, v in enumerate(vx):
        """
        sim = python_csdl_backend.Simulator(BEMRunModel(
        rotor_radius=rotor_radius,
        rpm=n,
        Vx=v,
        altitude=altitude,
        shape=shape,
        num_blades=num_blades,
        airfoil_name='NACA_4412',
        airfoil_polar=airfoil_polar,
        chord_distribution=chord,
        twist_distribution=twist,
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,))
        """

        sim = python_csdl_backend.Simulator(BILDRunModel(
        rotor_radius=rotor_radius,
        reference_chord=0.1,
        reference_radius=rotor_radius,
        rpm=n,
        Vx=v,
        altitude=altitude,
        shape=shape,
        num_blades=num_blades,
        airfoil_name='NACA_4412',
        airfoil_polar=airfoil_polar,
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,))

        sim.run()

        datact[i,j] = sim['C_T'].flatten()
        datacp[i,j] = sim['C_P'].flatten()

levels = np.arange(-0.5,0.5,0.03)

plt.contourf(rpm,vx,datact,)
plt.colorbar(shrink=1)
plt.show()

plt.contourf(rpm,vx,datacp)
plt.colorbar(shrink=1)
plt.show()
