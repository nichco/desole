from smt.surrogate_models import RBF
import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


n = 1000
x_lim = 3000.0 # (m)
be = 10
pi = 300
pf = 3000
bf = pf + 500
obs_height = 100
x = np.linspace(0,x_lim,n)
obs = np.zeros((n))

for i in range(0,n):
    xi = x[i]
    if xi > pi and xi <= pf:
        obs[i] = obs_height
    elif xi > be and xi < pi:
        obs[i] = (np.sin((np.pi/(pi-be))*(xi-be)-(np.pi/2)) + 1)*(obs_height/2)
    elif xi > pf and xi < bf:
        obs[i] = (np.sin((np.pi/(bf-pf))*(xi-bf)-(np.pi/2)) + 1)*(obs_height/2)


sm_obs = RBF(d0=50,print_global=False,print_solver=False,)
sm_obs.set_training_values(x, obs)
sm_obs.train()




num = 100
x_p = np.linspace(0,x_lim,num)
obs_p = sm_obs.predict_values(x_p)

print(np.array2string(x_p.flatten(),separator=','))
print(np.array2string(obs_p.flatten(),separator=','))

plt.plot(x_p,obs_p)
#plt.scatter(x,obs)
plt.xlim(0,500)
plt.show()

exit()



class Obs(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']
        x = self.declare_variable('x', shape=n)

        # custom operation insertion
        obsi = csdl.custom(x, op=ObsExplicit(num_nodes=n))
        self.register_output('obsi', obsi)

class ObsExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        # input:
        self.add_input('x', shape=n)

        # output:
        self.add_output('obsi', shape=n)

        self.declare_derivatives('obsi', 'x')

    def compute(self, inputs, outputs):
        n = self.parameters['num_nodes']

        # surrogate model
        obsi = np.zeros((n))
        for i in range(n):
            xi = np.array([inputs['x'][i]])
            obsi[i] = sm_obs.predict_values(xi)

        outputs['obsi'] = 1*obsi

    def compute_derivatives(self, inputs, derivatives):
        n = self.parameters['num_nodes']

        dobsi_dxi = np.zeros((n))
        for i in range(n):
            xi = np.array([inputs['x'][i]])
            dobsi_dxi[i] = sm_obs.predict_derivatives(xi, 0)

        derivatives['obsi', 'x'] = np.diag(dobsi_dxi)


if __name__ == '__main__':
    # run model
    num = 100
    sim = python_csdl_backend.Simulator(Obs(num_nodes=num))
    sim.run()
    print(sim['obsi'])

    # print partials
    sim.check_partials(step=1E-3,compact_print=True)