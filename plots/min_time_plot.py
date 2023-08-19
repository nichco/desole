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
dt = 0.39339627
time = np.arange(0, num*dt, dt)

x = np.array([0.00000000e+00,4.71875085e-01,1.81519407e+00,4.03433318e+00,
 7.13237535e+00,1.11129150e+01,1.59798766e+01,2.17376316e+01,
 2.83904591e+01,3.59424962e+01,4.43975009e+01,5.37588209e+01,
 6.40293835e+01,7.52117627e+01,8.73075699e+01,1.00317790e+02,
 1.14243110e+02,1.29083217e+02,1.44837747e+02,1.61506237e+02,
 1.79081486e+02,1.97539599e+02,2.16813073e+02,2.36763054e+02,
 2.57212484e+02,2.77968148e+02,2.98833023e+02,3.19636381e+02,
 3.40322655e+02,3.60931042e+02,3.81469574e+02,4.01949959e+02,
 4.22406116e+02,4.42903625e+02,4.63534514e+02,4.84427785e+02,
 5.05855153e+02,5.27905619e+02,5.50582920e+02,5.73880367e+02])
z = np.array([300.        ,299.85665968,299.4354648 ,298.7389737 ,297.7677152 ,
 296.5240955 ,295.01192479,293.23659747,291.20412996,288.92031249,
 286.39088761,283.62117237,280.61561538,277.37846924,273.91256626,
 270.21982988,266.3022617 ,262.16037774,257.80950378,253.34121796,
 248.92808789,244.79890183,241.44366888,239.21830848,238.28367342,
 238.72933491,240.58971825,243.80256922,248.07444051,253.05326146,
 258.57357309,264.50803926,270.71742447,277.0274583 ,283.23560805,
 289.05836847,293.81774928,297.24850947,299.31165524,299.99996125]) - 300
a = np.array([-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,
 -0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,
 -0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,
 -0.33534345,-0.30123916,-0.27200789,-0.17364556,-0.09018626,-0.02203407,
  0.04637975, 0.11267852, 0.17669631, 0.20350037, 0.19978383, 0.21567508,
  0.22528214, 0.22799854, 0.22057962, 0.19652171, 0.17666258, 0.05030631,
 -0.02285031,-0.08103411,-0.13621113,-0.0361419])




# interpolate the data
num = 7
x_prime, y_prime, a_prime, t_prime = interp_data(x=x, y=z, a=a, t=time, num=num)


fig = plt.figure(figsize=(7,3))
plt.xlim([x[0],x[-1]])
plt.ylim([-300,300])
fontsize = 16

plt.axhline(y=0, color='lavender', linestyle='dashed', linewidth=2, alpha=1, zorder=0, label = '_nolegend_')

for i in range(num):
    coef = 1.75
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + coef*np.rad2deg(a_prime[i])))
    plt.scatter(x_prime[i], y_prime[i], marker=new_marker, s=5000, c='mistyrose', zorder=3, edgecolor='black', linewidth=1, alpha=1)

plt.plot(x, z, c='red', alpha=1, linewidth=3, zorder=4)
# plt.scatter(x, z, marker='o', s=30, c='white', zorder=4, edgecolor='black')

plt.xlabel('Horizontal Position (m)', fontsize=fontsize)
plt.ylabel('Altitude (m)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.savefig('min_time.pdf', transparent=True, bbox_inches="tight")
plt.show()