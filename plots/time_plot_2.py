import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"


motor = 0.320767147999959
aero = 0.01248979999999733
prop = 0.02558119999594055

motor_s = 0.001592399989021942
prop_s = 0.0027942000015173107
aero_s = 0.00012549999519251287

species = ("Motor Model", "Rotor Model", "VLM Model")
penguin_means = {
    'Physics-Based Model': (motor, prop, aero),
    'Surrogate': (motor_s, prop_s, aero_s),
}

c1 = 'royalblue'
c2 = 'orange'
colors = [c1,c2,c1,c2,c1,c2]

ec1 = 'blue'
ec2 = 'darkorange'

ec = [ec1,ec2,ec1,ec2,ec1,ec2]


h1 = '//'
h2 = '\\\\'
hatch = [h1,h2,h1,h2,h1,h2]

x = np.arange(len(species))  # the label locations
width = 0.375  # the width of the bars
multiplier = 0.5
ep = 0.01
eps = [-ep,ep,-ep,ep,-ep,ep]

fig = plt.figure(figsize=(7,3))
fontsize = 16

i = 0
for attribute, measurement in penguin_means.items():
    offset = width * multiplier + eps[i]
    rects = plt.bar(x + offset, measurement, width, label=attribute, log=True, zorder=2, color=colors[i], edgecolor=ec[i], hatch=hatch[i])
    # plt.bar_label(rects, padding=3)
    multiplier += 1
    i += 1


plt.grid(color='lavender', zorder=0, axis='y')

plt.ylabel('Model Evaluation Time (s)', fontsize=fontsize)

plt.xticks(x + width, species, fontsize=fontsize-2)
plt.legend(loc='upper right', ncols=2, fontsize=fontsize-2, frameon=False)


plt.savefig('time.pdf', transparent=True, bbox_inches="tight")
plt.show()