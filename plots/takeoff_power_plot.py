import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



num = 40
# dt = 1.96752491
dt = 1.55066567
time = np.arange(0, num*dt, dt)

cruise_power = np.array([455604.43805601,459269.95528883,459451.13150743,459284.08840977,
 459137.03798569,458425.08526096,456163.3386993 ,451404.3106722 ,
 430302.97889327,206936.04388849,208511.36705102,182811.297018  ,
 182492.57777968,179732.41438006,179710.41562201,180909.87477914,
 178557.76132935,180518.20017443,183908.34135807,186135.6822146 ,
 187573.05729361,188245.88412676,188628.06535522,188582.76891069,
 189072.05629223,189168.71279717,189341.98921454,189627.35504872,
 189735.43663184,189878.60759844,190033.77217496,190092.38084612,
 190274.7622248 ,190515.55673806,190276.56838776,190069.902016  ,
 187499.00410128,180884.01490766,167153.06231148,158946.23747162])

lift_power = np.array([1.63636453e+05,1.65314920e+05,1.62046453e+05,1.58072705e+05,
 1.52837392e+05,5.98120472e+04,2.94463815e+04,1.19583267e+04,
 2.61304029e+02,3.36165588e-37,3.52414434e-37,0.00000000e+00,
 0.00000000e+00,0.00000000e+00,3.01316436e-07,1.94041264e-05,
 7.19259689e-07,9.96028112e-38,5.68003296e-13,1.93876359e-12,
 0.00000000e+00,9.26301149e-13,3.65029578e-10,3.58737559e-10,
 2.35338384e-09,8.31018103e-09,8.63735290e-09,1.23757588e-08,
 2.57445492e-08,4.63546210e-08,3.37087770e-08,4.44093741e-08,
 1.99670736e-08,1.03860076e-08,4.03176999e-10,4.01474258e-12,
 1.18532611e-12,6.79306906e-12,4.78568115e-36,4.57473006e-08])



fig = plt.figure(figsize=(7,3))
plt.xlim([0,time[-1]])
plt.ylim([0,500000*1E-3])
fontsize = 16


plt.axhline(y=468300*1E-3, color='black', linestyle='dashed', linewidth=2, alpha=0.65, zorder=1)
plt.axhline(y=165000*1E-3, color='black', linestyle='dotted', linewidth=2, alpha=0.65, zorder=1)

plt.plot(time, cruise_power*1E-3, c='royalblue', alpha=1, linewidth=3, zorder=4)
plt.plot(time, lift_power*1E-3, c='darkorange', alpha=1, linewidth=3, zorder=4)

x_p = np.linspace(time[0],time[-1],num)
plt.fill_between(x_p, 0, cruise_power*1E-3, alpha=0.75, color='lightblue',zorder=0)
plt.fill_between(x_p, 0, lift_power*1E-3, alpha=0.75, color='bisque',zorder=0)

plt.legend(['max cruise power constraint', 'max lift power constraint', 'cruise rotor power', 'lift rotor power'], frameon=False, fontsize=fontsize - 2, loc='center', bbox_to_anchor=(0.72, 0.65))

plt.xlabel('Time (s)', fontsize=fontsize)
plt.ylabel('Power (kW)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.savefig('power_takeoff.png', transparent=True, bbox_inches="tight", dpi=500)
plt.show()