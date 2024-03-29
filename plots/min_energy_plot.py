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
dt = 1.68061329
e = 11.4
time = np.arange(0, num*dt, dt)

x = np.array([   0.        ,   7.40047841,  28.52834412,  62.64472609, 109.32678126,
  169.76486076, 246.57342672, 338.15704828, 442.38121739, 554.13706133,
  668.35707761, 782.91204035, 897.2385848 ,1011.22815756,1125.00177113,
 1238.56370956,1351.8938344 ,1464.98525838,1577.84408385,1690.48934536,
 1802.95138668,1915.26735696,2027.47718121,2139.61762389,2251.71817257,
 2363.79794681,2475.86266329,2587.90217977,2699.8873586 ,2811.76486968,
 2923.45022835,3034.81830295,3145.69119873,3255.82362102,3364.88530634,
 3472.44378832,3577.97267762,3680.98458341,3781.66706786,3881.88599311])
z = np.array([300.        ,295.1504906 ,278.21904913,248.98602771,210.92436291,
 169.60414569,130.83157201, 95.78815083, 71.44742021, 61.50830629,
  62.52046409, 67.99489646, 75.26415027, 82.90610795, 90.15944009,
  97.44521368,104.81769521,112.23107412,119.62856205,126.95289638,
 134.1550756 ,141.20465055,148.08973764,154.82351088,161.43689708,
 167.97448424,174.49338482,181.06065439,187.75647685,194.68109709,
 201.9602869 ,209.75252669,218.25375932,227.69852143,238.35706255,
 250.51486546,264.35634101,279.56414637,293.61929974,300.00027777]) - 300
a = np.array([-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,-0.34906585,
 -0.34906585,-0.22944776,-0.09006426, 0.01702444, 0.04831819, 0.06324071,
  0.06843118, 0.06202238, 0.06385837, 0.06535045, 0.06637646, 0.06690377,
  0.06689903, 0.06635754, 0.06546876, 0.06432984, 0.06322013, 0.06229487,
  0.06169151, 0.0615609 , 0.06201306, 0.06318554, 0.06530946, 0.06866701,
  0.07364442, 0.08071966, 0.09044155, 0.10346209, 0.12045967, 0.14188127,
  0.16492294, 0.17878315, 0.11591539,-0.03386699])




# interpolate the data
num = 7
x_prime, y_prime, a_prime, t_prime = interp_data(x=x, y=z, a=a, t=time, num=num)


fig = plt.figure(figsize=(7,3))
plt.xlim([x[0],x[-1]])
plt.ylim([-400,400])
fontsize = 16

plt.axhline(y=0, color='lavender', linestyle='dashed', linewidth=2, alpha=1, zorder=0, label = '_nolegend_')

for i in range(num):
    coef = 1.75
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + coef*np.rad2deg(a_prime[i])))
    plt.scatter(x_prime[i], y_prime[i], marker=new_marker, s=5000, c='lightblue', zorder=3, edgecolor='black', linewidth=1, alpha=1)

plt.plot(x, z, c='blue', alpha=1, linewidth=3, zorder=4)
# plt.scatter(x, z, marker='o', s=30, c='white', zorder=4, edgecolor='black')

plt.xlabel('Horizontal Position (m)', fontsize=fontsize)
plt.ylabel('Altitude (m)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.savefig('min_energy.pdf', transparent=True, bbox_inches="tight")
plt.show()