import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import numpy as np
from plot import interp_data, marker

plt.rcParams["font.family"] = "Times New Roman"


# trajectory data
num = 40
dt = 1.82811623
time = np.arange(0, num*dt, dt)

x = np.array([   0.        ,   3.00177723,  13.64516195,  33.20738622,  59.28874176,
   87.81001435, 115.06604101, 140.15187418, 164.84392325, 190.18576302,
  217.62541447, 249.74320247, 291.61002884, 345.90028514, 407.48006754,
  475.46031333, 549.94603608, 630.24166856, 714.96386567, 803.72790832,
  896.37860256, 992.85916111,1093.16015323,1197.26537735,1305.23755452,
 1416.58652046,1530.17134924,1644.87309011,1759.93800035,1875.07812605,
 1990.25116388,2105.43780955,2220.5910357 ,2335.61075306,2450.23264794,
 2563.87668754,2675.71746271,2784.94643742,2891.96163963,2999.99996602])
z = np.array([  0.        ,  0.83125521,  2.81189907,  5.49207574, 10.02937486,
  17.84044697, 28.9488218 , 41.90027168, 55.24425452, 68.5145916 ,
  81.29531796, 92.68382832, 99.69191016, 99.91475667, 99.9151115 ,
  99.91462563, 99.91546428, 99.91702206, 99.92432817, 99.92493632,
  99.92728342, 99.9280034 , 99.94728131, 99.93086118, 99.95549516,
 102.46357433,108.859386  ,118.70931444,130.94721589,144.21075904,
 157.73165576,171.36878406,185.17889345,199.34461428,214.40974234,
 231.28367989,250.54679571,271.93364814,291.60298045,299.99996481])
a = np.array([ 0.12492133, 0.02952486,-0.01895811, 0.06973965, 0.18792869, 0.30309727,
  0.34906585, 0.29414646, 0.28210215, 0.2417602 , 0.18318552, 0.02383716,
 -0.17371992, 0.15069567, 0.15129883, 0.16074984, 0.14558076, 0.11546069,
  0.09604475, 0.08014864, 0.06720266, 0.05440823, 0.04697408, 0.02844422,
  0.04217141, 0.07350207, 0.10041125, 0.1212363 , 0.1294627 , 0.13060295,
  0.13129713, 0.1327248 , 0.1352967 , 0.14183717, 0.15871762, 0.18448351,
  0.21329906, 0.23459098, 0.13656447,-0.03929315])



x_p = np.array([   0.        ,  30.3030303 ,  60.60606061,  90.90909091, 121.21212121,
  151.51515152, 181.81818182, 212.12121212, 242.42424242, 272.72727273,
  303.03030303, 333.33333333, 363.63636364, 393.93939394, 424.24242424,
  454.54545455, 484.84848485, 515.15151515, 545.45454545, 575.75757576,
  606.06060606, 636.36363636, 666.66666667, 696.96969697, 727.27272727,
  757.57575758, 787.87878788, 818.18181818, 848.48484848, 878.78787879,
  909.09090909, 939.39393939, 969.6969697 ,1000.        ,1030.3030303 ,
 1060.60606061,1090.90909091,1121.21212121,1151.51515152,1181.81818182,
 1212.12121212,1242.42424242,1272.72727273,1303.03030303,1333.33333333,
 1363.63636364,1393.93939394,1424.24242424,1454.54545455,1484.84848485,
 1515.15151515,1545.45454545,1575.75757576,1606.06060606,1636.36363636,
 1666.66666667,1696.96969697,1727.27272727,1757.57575758,1787.87878788,
 1818.18181818,1848.48484848,1878.78787879,1909.09090909,1939.39393939,
 1969.6969697 ,2000.        ,2030.3030303 ,2060.60606061,2090.90909091,
 2121.21212121,2151.51515152,2181.81818182,2212.12121212,2242.42424242,
 2272.72727273,2303.03030303,2333.33333333,2363.63636364,2393.93939394,
 2424.24242424,2454.54545455,2484.84848485,2515.15151515,2545.45454545,
 2575.75757576,2606.06060606,2636.36363636,2666.66666667,2696.96969697,
 2727.27272727,2757.57575758,2787.87878788,2818.18181818,2848.48484848,
 2878.78787879,2909.09090909,2939.39393939,2969.6969697 ,3000.        ])

obs_p = np.array([-1.98370218e-03, 1.20453933e+00, 7.32769578e+00, 1.80076898e+01,
  3.21045232e+01, 4.81125810e+01, 6.43221182e+01, 7.90018474e+01,
  9.05837924e+01, 9.78308363e+01, 1.00015866e+02, 1.00005566e+02,
  1.00002875e+02, 1.00001775e+02, 1.00001173e+02, 1.00000797e+02,
  1.00000547e+02, 1.00000378e+02, 1.00000261e+02, 1.00000180e+02,
  1.00000124e+02, 1.00000086e+02, 1.00000059e+02, 1.00000040e+02,
  1.00000028e+02, 1.00000019e+02, 1.00000013e+02, 1.00000009e+02,
  1.00000006e+02, 1.00000004e+02, 1.00000003e+02, 1.00000002e+02,
  1.00000001e+02, 1.00000001e+02, 1.00000001e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
  1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 9.99999999e+01,
  9.99999999e+01, 9.99999999e+01, 9.99999999e+01, 9.99999998e+01,
  9.99999997e+01, 9.99999996e+01, 9.99999995e+01, 9.99999993e+01,
  9.99999991e+01, 9.99999987e+01, 9.99999982e+01, 9.99999972e+01,
  9.99999949e+01, 9.99999884e+01, 9.99999728e+01, 9.99999675e+01])



# interpolate the data
num = 6
x_prime, y_prime, a_prime, t_prime = interp_data(x=x, y=z, a=a, t=time, num=num)


fig = plt.figure(figsize=(7,3))
plt.xlim([x[0],x[-1]])
plt.ylim([-100,500])
fontsize = 16

plt.axhline(y=300, color='black', linestyle='dashed', linewidth=2, alpha=0.5, zorder=1)

for i in range(num):
    coef = 2
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + coef*np.rad2deg(a_prime[i])))
    plt.scatter(x_prime[i], y_prime[i], marker=new_marker, s=5000, c='palegreen', zorder=3, edgecolor='black', linewidth=1, alpha=1, label = '_nolegend_')

plt.plot(x, z, c='limegreen', alpha=1, linewidth=3, zorder=11, label = '_nolegend_')


# x_p = np.linspace(x[0],x[-1],num)
plt.fill_between(x_p, -100, obs_p, alpha=0.375, color='mistyrose', hatch='//', edgecolor='red', zorder=10)
plt.plot(x_p, obs_p, c='red', alpha=0.5, linewidth=1, zorder=12, label = '_nolegend_')

plt.xlabel('Horizontal Position (m)', fontsize=fontsize)
plt.ylabel('Altitude (m)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.legend(['target altitude', 'no-fly zone'], frameon=False, fontsize=fontsize - 2, loc='upper left')

plt.text(2500,400,'58.7 MJ',fontsize=fontsize-2,c='black',bbox=dict(facecolor='white',alpha=1,edgecolor='black',boxstyle='round'))

plt.savefig('nfz.pdf', transparent=True, bbox_inches="tight")
plt.show()