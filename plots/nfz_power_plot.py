import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



num = 40
dt = 1.82811623
time = np.arange(0, num*dt, dt)

cruise_power = np.array([466486.06134283,466830.49118527,466990.65628615,467194.89019238,
 467281.69633466,467275.0909368 ,467189.74361539,467062.96496927,
 466961.82327401,466841.39831302,466702.24627274,466554.56173961,
 466574.14578889,466601.44257931,466334.80155897,465804.80524188,
 376798.88945754,226555.49176937,211736.22550446,195718.89259366,
 192768.60204862,192413.54422147,194508.62732281,197650.3912454 ,
 202647.60638669,206952.05324456,209724.44286935,211782.20845711,
 212708.27218827,212989.33107492,213328.70858144,213567.87300984,
 213870.07964923,214147.53894079,214297.33079673,213315.10863051,
 209589.90881096,201212.39264519,186021.96915437,178968.67721409])
lift_power = np.array([1.68824941e+05,1.69079280e+05,1.68969645e+05,1.68894476e+05,
 1.68845195e+05,1.68800265e+05,1.68750749e+05,1.68687016e+05,
 1.68600909e+05,1.68475330e+05,1.68287306e+05,1.67986383e+05,
 1.67424113e+05,4.95612722e+04,2.32585647e+04,7.60356452e+03,
 1.07109984e+02,0.00000000e+00,5.75196979e-14,2.22650467e-38,
 9.37313073e-17,6.64035170e-19,5.42849371e-22,2.74976060e-17,
 3.42017892e-17,3.28977164e-17,2.20440901e-16,6.12577085e-16,
 1.41573314e-15,2.07983295e-15,1.99938195e-15,1.95705011e-15,
 1.87263996e-15,1.22213305e-15,5.73904068e-16,2.37127097e-16,
 6.11940915e-17,3.36393406e-16,3.26713371e-14,3.07285734e-13])



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

plt.savefig('power_nfz.pdf', transparent=True, bbox_inches="tight")
plt.show()