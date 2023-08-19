import numpy as np
import matplotlib.pyplot as plt


dt = 1.8505198

ux = np.array([1478.60216861,1498.31470337,1515.70669826,1534.32788225,1554.93035665,
 1577.20977083,1592.47574373,1603.27247772,1611.10551639,1451.41269526,
 1166.83243067,1082.32234382,1011.41409786, 968.14123336, 934.16988391,
  908.49280534, 887.75305781, 870.55302742, 856.79617618, 845.33712342,
  835.54381438, 827.03265171, 819.85871236, 813.40770417, 807.65611136,
  802.90376728, 798.43370765, 794.47961623, 791.03986669, 787.99380008,
  785.27969353, 782.65401116, 780.41556355, 778.34655732, 776.50900531,
  774.75744928, 773.29783687, 771.97005393, 770.99516333, 769.71474992])
uz = np.array([1.19914498e+03,1.20314416e+03,1.19522943e+03,1.18535395e+03,
 1.17380671e+03,1.15710198e+03,7.83996493e+02,6.54238417e+02,
 4.48560261e+02,2.08129749e+02,4.35765726e-12,6.08813276e-04,
 6.90743780e-03,3.92150890e-02,1.17052983e-06,4.11456658e-03,
 2.92720776e-03,4.04768543e-03,9.72088104e-03,3.13699615e-03,
 1.19664319e-03,1.72548313e-04,4.52949676e-04,9.82175912e-04,
 4.08037759e-03,7.88449886e-04,1.11280376e-03,3.09799611e-03,
 6.43624782e-04,4.94105147e-04,4.72993706e-04,3.61063450e-04,
 8.34396556e-05,1.10791577e-04,4.69237046e-04,7.66395555e-04,
 1.55558068e-03,6.24237645e-04,1.80709496e-03,5.07879455e-03])
ua = np.array([-0.01350389,-0.03104296,-0.02994083,-0.03743226,-0.04973968,-0.03845421,
  0.14958518, 0.15123054, 0.16194037, 0.15171894, 0.13355846, 0.11829502,
  0.10777767, 0.09956571, 0.09286108, 0.08716222, 0.08222364, 0.07782967,
  0.07388016, 0.07033121, 0.06698612, 0.06397175, 0.06110693, 0.05842687,
  0.05596889, 0.05354298, 0.05130862, 0.04920144, 0.04709851, 0.04529791,
  0.04323155, 0.04161012, 0.03976235, 0.03811269, 0.03656443, 0.03495403,
  0.03365892, 0.03154088, 0.03183971, 0.02871176])
x = np.array([   0.        ,   4.93363012,  19.76714062,  44.74531333,  80.1972258 ,
  126.44611437, 182.40485269, 245.08427937, 314.0702928 , 389.23950401,
  469.29365755, 552.41396744, 637.85569762, 725.22633616, 814.27304883,
  904.81513642, 996.71647165,1089.8693221 ,1184.18605183,1279.59608939,
 1376.04034248,1473.46897854,1571.83812994,1671.11017407,1771.2515061 ,
 1872.23125725,1974.0236218 ,2076.60439152,2179.95100355,2284.04401685,
 2388.86397317,2494.39601408,2600.62332485,2707.53224088,2815.10993615,
 2923.34298196,3032.21940991,3141.7274905 ,3251.86191901,3362.59402091])
z = np.array([300.        ,299.94943048,299.90583684,299.90694906,299.91001999,
 299.91137664,299.91299863,299.92218962,299.91676936,299.91803478,
 299.91944718,299.92384732,299.92640579,299.92902431,299.93035724,
 299.93146003,299.93238597,299.9336559 ,299.93412458,299.93467754,
 299.9362425 ,299.93628653,299.93754246,299.93766159,299.93783212,
 299.93942308,299.93841005,299.9383225 ,299.93864399,299.9388921 ,
 299.94289783,299.94021248,299.94332509,299.94066478,299.93898168,
 299.9394884 ,299.94183879,299.93943267,299.9398359 ,300.00000002])
v = np.array([ 0.1       , 5.3022081 ,10.73720863,16.29019149,22.0690022 ,27.88438576,
 32.21342415,35.55593472,39.01005136,42.10952994,44.20750424,45.58439112,
 46.72177344,47.68549926,48.53793597,49.30594191,50.00933639,50.66027193,
 51.26886802,51.84282893,52.38752991,52.90714018,53.40468989,53.88328524,
 54.34434897,54.79006344,55.22261324,55.64229697,56.0507359 ,56.44872723,
 56.83720138,57.21771826,57.58926276,57.95429918,58.31199193,58.662862  ,
 59.00693003,59.34743188,59.67905967,60.        ])





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




plt.plot(cruise_power)
plt.show()