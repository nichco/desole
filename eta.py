import numpy as np




def efficiency(e,dh,dv,m,g):
    ke = 0.5*m*dv**2
    pe = 0.5*m*g*dh
    return (ke + pe)/e




dh = 0
dv = 60
e = 37.9E6
m = 3000
g = 9.81

eta = efficiency(e,dh,dv,m,g)
print('eta: ', eta)