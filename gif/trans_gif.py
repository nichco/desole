from gif import create_frame, combine_frames, interp_data
import numpy as np
import imageio


num = 40
dt = 1.8505198
time = np.arange(0, num*dt, dt)

x = np.array([   0.        ,   4.93363012,  19.76714062,  44.74531333,  80.1972258 ,126.44611437, 182.40485269, 245.08427937, 314.0702928 , 389.23950401,
  469.29365755, 552.41396744, 637.85569762, 725.22633616, 814.27304883,904.81513642, 996.71647165,1089.8693221 ,1184.18605183,1279.59608939,
 1376.04034248,1473.46897854,1571.83812994,1671.11017407,1771.2515061 ,1872.23125725,1974.0236218 ,2076.60439152,2179.95100355,2284.04401685,
 2388.86397317,2494.39601408,2600.62332485,2707.53224088,2815.10993615,2923.34298196,3032.21940991,3141.7274905 ,3251.86191901,3362.59402091])
z = np.array([300.        ,299.94943048,299.90583684,299.90694906,299.91001999,299.91137664,299.91299863,299.92218962,299.91676936,299.91803478,
 299.91944718,299.92384732,299.92640579,299.92902431,299.93035724,299.93146003,299.93238597,299.9336559 ,299.93412458,299.93467754,
 299.9362425 ,299.93628653,299.93754246,299.93766159,299.93783212,299.93942308,299.93841005,299.9383225 ,299.93864399,299.9388921 ,
 299.94289783,299.94021248,299.94332509,299.94066478,299.93898168,299.9394884 ,299.94183879,299.93943267,299.9398359 ,300.00000002]) - 300
a = np.array([-0.01350389,-0.03104296,-0.02994083,-0.03743226,-0.04973968,-0.03845421,0.14958518, 0.15123054, 0.16194037, 0.15171894, 0.13355846, 0.11829502,
  0.10777767, 0.09956571, 0.09286108, 0.08716222, 0.08222364, 0.07782967,0.07388016, 0.07033121, 0.06698612, 0.06397175, 0.06110693, 0.05842687,
  0.05596889, 0.05354298, 0.05130862, 0.04920144, 0.04709851, 0.04529791,0.04323155, 0.04161012, 0.03976235, 0.03811269, 0.03656443, 0.03495403,0.03365892, 0.03154088, 0.03183971, 0.02871176])


x_prime, y_prime, a_prime, t_prime = interp_data(x=x, y=z, a=a, t=time, num=300)

for i, t in enumerate(t_prime):
    create_frame(x=x_prime,
                 y=y_prime, 
                 a=1.75*np.rad2deg(a_prime[i]), 
                 i=i,
                 s=10000,
                 figsize=(8,3),
                 xlim=[0,4200],
                 ylim=[-400,400], 
                 xlabel='Horizontal Position (m)', 
                 ylabel='Altitude (m)',
                 title='Min-Energy Transition',
                 fontsize=16,
                 marker_color='blue')

frames = combine_frames(t_prime)

imageio.mimsave('constant_altitude_min_energy.gif', frames, fps = 60)