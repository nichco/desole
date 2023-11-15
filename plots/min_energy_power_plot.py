import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



num = 40
dt = 1.66563477
time = np.arange(0, num*dt, dt)

cruise_power = np.array([463521.61932622,465873.05744937,385036.88207661,186191.66298988,
  89118.05195625, 61401.94880017, 79413.9694206 , 91214.91187599,
  95643.54508214, 97632.97780825, 98962.11777356,100474.47599167,
 101072.48733547,101386.08738847,101497.56728405,101590.49894157,
 101676.13787234,101767.38810814,101882.40773938,102006.22054507,
 102129.94996316,102255.79608835,102373.89860756,102420.18533796,
 102532.57920741,102785.5768867 ,102573.7829716 ,102309.78387369,
 102818.45781545,103158.10497721,103271.34717791,103064.87039264,
 103301.98029657,103250.19743863,102618.13260158,103439.26111779,
 100465.59573077, 97151.13102479, 89572.79417269, 84330.68250944])

lift_power = np.array([1.29107303e+05,8.13645289e+04,4.12029831e+04,1.59642279e+04,
 2.64718878e+03,0.00000000e+00,6.81840801e-09,6.43851307e-46,
 9.27364147e-45,7.80940983e-44,1.08544766e-13,5.14775450e-14,
 2.06398453e-10,9.07531518e-12,2.65428512e-12,5.13099025e-11,
 5.74238648e-10,1.52853206e-09,3.75598543e-09,3.15983412e-09,
 1.33322965e-09,9.80008011e-12,6.48149134e-46,0.00000000e+00,
 0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
 0.00000000e+00,0.00000000e+00,2.58596856e-46,4.52679375e-47,
 0.00000000e+00,0.00000000e+00,3.35016878e-46,9.15938676e-12,
 5.18414455e-10,2.16442521e-44,5.57270667e-46,0.00000000e+00])



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

plt.legend(['max cruise power constraint', 'max lift power constraint', 'cruise rotor power', 'lift rotor power'], frameon=False, fontsize=fontsize - 2, loc='center', bbox_to_anchor=(0.72, 0.6))

plt.xlabel('Time (s)', fontsize=fontsize)
plt.ylabel('Power (kW)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.savefig('power_energy.png', transparent=True, bbox_inches="tight", dpi=500)
plt.show()