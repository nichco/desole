import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



motor = 0.320767147999959
aero = 0.01248979999999733
prop = 0.02558119999594055

motor_s = 0.001592399989021942
prop_s = 0.0027942000015173107
aero_s = 0.00012549999519251287


val = ['Motor Model', 'Motor Surrogate', 'Rotor Model', 'Rotor Surrogate', 'VLM Model', 'VLM Surrogate']
time = [motor, motor_s, prop, prop_s, aero, aero_s]

c1 = 'darkgray'
c2 = 'royalblue'
colors = [c1,c2,c1,c2,c1,c2]

ec1 = 'dimgray'
ec2 = 'darkblue'

ec = [ec1,ec2,ec1,ec2,ec1,ec2]


fig = plt.figure(figsize=(4,1.75))
fontsize = 16

plt.bar(val, time, color=colors, hatch='//', width=0.6, edgecolor=ec, log=True, zorder=2)

plt.xticks(rotation=16)

plt.grid(color='lavender', zorder=0, axis='y')

plt.ylabel('Model Evaluation Time (s)')


plt.savefig('time.pdf', transparent=True, bbox_inches="tight")
plt.show()