import numpy as np

e = 31.69143046 # MJ
dt = 1.55066567
ux = np.array([ 297.42441667,1489.88123097,1508.24341523,1527.83267565,1548.58672461,
 1570.57871114,1585.99458478,1597.95551439,1610.53816477,1623.95359469,
 1535.52611321,1381.23666564,1336.58401601,1320.16665937,1326.70209958,
 1318.51299301,1304.28127438,1335.62394052,1351.55514573,1363.46536982,
 1376.53370036,1382.37702981,1388.94443822,1391.25101893,1390.90517663,
 1395.37920021,1395.08651094,1395.27380675,1397.3569207 ,1399.00677657,
 1399.45171053,1400.68495184,1399.91634876,1393.78238598,1370.09155577,
 1326.72841316,1257.48109828,1172.84730349,1084.90196519,1039.48003417])
uz = np.array([1.20192699e+03,1.20155499e+03,1.19918280e+03,1.19729297e+03,
 1.19494128e+03,1.19304245e+03,7.98488061e+02,6.79992609e+02,
 5.26639838e+02,3.25929774e+02,2.25972863e+01,0.00000000e+00,
 6.01444924e-01,3.60733447e-09,0.00000000e+00,2.00495612e-01,
 2.70317055e+00,1.49419092e-08,0.00000000e+00,0.00000000e+00,
 9.89632426e-10,4.17550955e-10,9.78541479e-02,4.12884021e-01,
 5.64343047e-01,7.78486034e-01,7.17560835e-01,6.95972477e-01,
 8.33543240e-01,8.15561009e-01,9.82395254e-01,1.02269121e+00,
 5.52275311e-01,5.27806807e-01,4.22494030e-01,1.59408376e-01,
 3.50061274e-02,5.33593464e-10,6.54014245e-01,1.26361599e-08])
ua = np.array([-0.29573282,-0.08152178,-0.09497642,-0.08862308,-0.09352745,-0.07926495,
  0.14126896, 0.14720638, 0.15301506, 0.15331223, 0.12029893, 0.12764657,
  0.11354434, 0.05378197, 0.08151392, 0.128511  ,-0.03615918, 0.01887473,
  0.04543392, 0.05520885, 0.08520177, 0.10676553, 0.12929847, 0.13204522,
  0.11736247, 0.15174479, 0.12521626, 0.13689884, 0.13674035, 0.13878282,
  0.13978194, 0.15810832, 0.19571392, 0.24616613, 0.30115456, 0.34906585,
  0.34906585, 0.34906585, 0.22896209,-0.07539365])
x = np.array([   0.        ,   3.66229078,  15.26348585,  35.59117768,  64.71724862,
  102.721881  , 148.57460309, 199.50237605, 255.07802251, 315.46941971,
  380.80404883, 450.69976326, 523.96956703, 600.17283526, 679.63031394,
  761.92438582, 846.37285967, 934.28392887,1025.90422347,1120.33896389,
 1216.92004067,1314.87717585,1413.51204869,1512.31631205,1611.27613205,
 1710.46974418,1809.4431618 ,1908.58035824,2007.71149892,2106.80693234,
 2205.84063478,2304.70135445,2402.85242022,2499.28602167,2592.72228691,
 2681.93154139,2766.3759184 ,2846.58389511,2923.4438406 ,2999.99966277])
z = np.array([ 0.00000000e+00,-7.66548428e-02,-9.92798216e-02,-9.91878955e-02,
 -9.90328510e-02,-9.87931521e-02,-9.88289480e-02,-9.82475798e-02,
 -9.87217812e-02,-9.58775277e-02,-9.89696086e-02,-3.90752098e-02,
  1.21829372e+00, 2.29774570e+00, 2.02175097e+00, 4.06624029e+00,
  6.58699991e+00, 3.10010694e+00, 1.76530127e+00, 3.39537960e+00,
  7.33821334e+00, 1.42583235e+01, 2.37692172e+01, 3.52362104e+01,
  4.69608205e+01, 5.87141336e+01, 7.17793673e+01, 8.39282375e+01,
  9.64788127e+01, 1.09177976e+02, 1.22013775e+02, 1.35313982e+02,
  1.50467338e+02, 1.68894072e+02, 1.91215156e+02, 2.16958234e+02,
  2.44042734e+02, 2.69428928e+02, 2.90556149e+02, 2.99999998e+02])
# v = np.array()


