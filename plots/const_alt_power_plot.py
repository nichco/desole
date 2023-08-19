import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



num = 40
dt = 1.8505198
time = np.arange(0, num*dt, dt)

cruise_power = np.array([456550.46691018,458405.3436325 ,456520.42727815,454430.71755476,
 452278.93408042,450190.90631931,447958.80028315,444591.755539  ,
 438530.03142193,312305.15900157,159433.92235382,125831.3493305 ,
 101749.84440906, 88549.55750249, 79001.71226787, 72207.79397786,
  66982.21628501, 62819.39799488, 59579.00791612, 56938.3450906 ,
  54721.82695616, 52823.96303286, 51233.27939003, 49817.86371514,
  48564.07644975, 47516.44988597, 46539.80343338, 45671.72968514,
  44907.26188266, 44222.47604905, 43603.94647793, 43009.46466319,
  42487.3094954 , 41999.64234919, 41555.91152987, 41131.73349497,
  40759.48264913, 40411.14620122, 40124.37230243, 39794.24287277])

lift_power = np.array([1.63310915e+05,1.65136557e+05,1.62160621e+05,1.58586126e+05,
 1.54542305e+05,1.48201975e+05,4.44076699e+04,2.58092877e+04,
 8.27774129e+03,8.26575726e+02,7.60045669e-39,2.07641944e-14,
 3.03630310e-11,5.56212952e-09,1.48071525e-22,6.43729078e-12,
 2.31988398e-12,6.13871173e-12,8.50948981e-11,2.86179975e-12,
 1.58962585e-13,4.76887674e-16,8.63203425e-15,8.80640432e-14,
 6.31811208e-12,4.56105375e-14,1.28304003e-13,2.76990036e-12,
 2.48517681e-14,1.12494929e-14,9.87371679e-15,4.39404080e-15,
 5.42573116e-17,1.27076099e-16,9.65878814e-15,4.21028877e-14,
 3.52219424e-13,2.27733701e-14,5.52563318e-13,1.22767912e-11])



fig = plt.figure(figsize=(7,3))
plt.xlim([0,time[-1]])
plt.ylim([0,500000*1E-3])
fontsize = 16


plt.axhline(y=468300*1E-3, color='black', linestyle='dashed', linewidth=2, alpha=0.65, zorder=1)
plt.axhline(y=170000*1E-3, color='black', linestyle='dotted', linewidth=2, alpha=0.65, zorder=1)

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

plt.savefig('power_ca.pdf', transparent=True, bbox_inches="tight")
plt.show()