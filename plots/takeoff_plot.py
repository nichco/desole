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
dt = 1.55066567
time = np.arange(0, num*dt, dt)

x = np.array([   0.        ,   3.70767653,  14.83330027,  33.2794553 ,  59.1875978 ,
   92.68191434, 133.85317222, 181.81092407, 234.44448435, 291.54936522,
  353.26652022, 419.64052683, 490.05027207, 564.13072919, 641.72263655,
  722.75097714, 807.16743597, 895.01207347, 986.19501421,1080.84589224,
 1178.02017982,1276.11730431,1374.17266037,1471.99967772,1569.71818504,
 1667.40398912,1765.0542393 ,1862.65185452,1960.23370805,2057.8313361 ,
 2155.39869181,2252.91641138,2350.29359053,2447.25551148,2543.40118123,
 2638.02446215,2730.40568283,2820.32521794,2909.30716678,2999.99998981])
z = np.array([ 0.00000000e+00,-3.52956151e-02,-9.89582096e-02,-9.94164010e-02,
 -9.91026036e-02,-9.88757392e-02,-9.89824215e-02,-9.82623903e-02,
 -9.78108977e-02,-9.82806787e-02,-9.83566256e-02,-9.84295807e-02,
 -9.65285942e-02,-8.82874380e-02,-9.72427171e-02,-8.87194718e-02,
 -5.81958166e-02,-9.64727726e-02, 3.57337997e-02,-8.56796680e-02,
  4.72984148e+00, 1.49935227e+01, 2.87643148e+01, 4.37827322e+01,
  5.87980553e+01, 7.36116770e+01, 8.83851344e+01, 1.03187063e+02,
  1.17868115e+02, 1.32437861e+02, 1.47120171e+02, 1.61880406e+02,
  1.76973780e+02, 1.92987470e+02, 2.10418720e+02, 2.30072006e+02,
  2.51964413e+02, 2.74584868e+02, 2.92835938e+02, 3.00000001e+02])
a = np.array([-0.03234891,-0.05948366,-0.03822165,-0.04839695,-0.04849385,-0.05643033,
 -0.0400625 , 0.15947768, 0.14880757, 0.15937989, 0.14860771, 0.11889409,
  0.09716061, 0.08013434, 0.064904  , 0.05651382, 0.0365931 , 0.04448925,
  0.00147602, 0.06123806, 0.1169543 , 0.15289292, 0.16410739, 0.16199618,
  0.15939768, 0.15932349, 0.16077996, 0.15976433, 0.15749437, 0.16001521,
  0.16054217, 0.16236789, 0.17375403, 0.18740993, 0.2164426 , 0.24841505,
  0.27914863, 0.25295825, 0.1230756 ,-0.00211068])




# interpolate the data
num = 7
x_prime, y_prime, a_prime, t_prime = interp_data(x=x, y=z, a=a, t=time, num=num)


fig = plt.figure(figsize=(8,3))
plt.xlim([x[0],x[-1]])
plt.ylim([-100,500])
fontsize = 16

plt.axhline(y=300, color='black', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)

for i in range(num):
    coef = 2
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + coef*np.rad2deg(a_prime[i])))
    plt.scatter(x_prime[i], y_prime[i], marker=new_marker, s=5000, c='lightgray', zorder=3, edgecolor='gray', linewidth=1, alpha=1, label = '_nolegend_')

plt.plot(x, z, c='black', alpha=1, linewidth=1, zorder=4, label = '_nolegend_')
plt.scatter(x, z, marker='o', s=30, c='white', zorder=4, edgecolor='black', label = '_nolegend_')


x_p = np.linspace(x[0],x[-1],num)
plt.fill_between(x_p, 0, -100, alpha=0.25, color='mistyrose', hatch='//', edgecolor='indianred')

plt.xlabel('Horizontal Position (m)', fontsize=fontsize)
plt.ylabel('Altitude (m)', fontsize=fontsize)

plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)

plt.legend(['target altitude', 'min altitude constraint'], frameon=False, fontsize=fontsize - 2, loc='upper left')

plt.savefig('takeoff.pdf', transparent=True, bbox_inches="tight")
plt.show()