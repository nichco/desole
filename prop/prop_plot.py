import numpy as np
from smt.surrogate_models import RMTB
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})



ctarr = np.array([[ 0.50845263,  0.48114841,  0.46048069,  0.45263168,  0.44918747,  0.45263168, 0.46048069,  0.48114841,  0.50845263],
    [ 0.41469071,  0.37799377,  0.35409169,  0.33544928,  0.32626682,  0.33544928, 0.35409169,  0.37799377,  0.41469071],
    [ 0.35626874,  0.31375178,  0.27994125,  0.25443275,  0.24395818,  0.25443275, 0.27994125,  0.31375178,  0.35626874],
    [ 0.33795284,  0.29184094,  0.25599661,  0.2345676,   0.26461192,  0.2345676, 0.25599661,  0.29184094,  0.33795284],
    [ 0.34362491,  0.29646818,  0.26162671,  0.24395124,  0.24127203,  0.24395124, 0.26162671,  0.29646818,  0.34362491],
    [ 0.34468518,  0.29754716,  0.26353205,  0.24336877,  0.23664139,  0.24336877, 0.26353205,  0.29754716,  0.34468518],
    [ 0.31969465,  0.25751606,  0.21456789,  0.20389248,  0.20295964,  0.20389248, 0.21456789,  0.25751606,  0.31969465],
    [ 0.28051414,  0.20418248,  0.13734402,  0.10851712,  0.11667478,  0.10851712, 0.13734402,  0.20418248,  0.28051414],
    [ 0.23358927,  0.14279756,  0.0603329,   0.00172831, -0.0176337,   0.00172831, 0.0603329,   0.14279756,  0.23358927]])

cparr = np.array([[0.501836,   0.46373982, 0.43129937, 0.41990135, 0.41816146, 0.41990135, 0.43129937, 0.46373982, 0.501836  ],
    [0.41321428, 0.37064409, 0.33877887, 0.31762158, 0.30942568, 0.31762158, 0.33877887, 0.37064409, 0.41321428],
    [0.33611122, 0.29281586, 0.25239685, 0.22285395, 0.21143836, 0.22285395, 0.25239685, 0.29281586, 0.33611122],
    [0.28741357, 0.23773751, 0.19594548, 0.17230629, 0.24126226, 0.17230629, 0.19594548, 0.23773751, 0.28741357],
    [0.28558827, 0.23334922, 0.19459788, 0.17535239, 0.170538,   0.17535239, 0.19459788, 0.23334922, 0.28558827],
    [0.31958289, 0.27173158, 0.23914474, 0.22150391, 0.2156555,  0.22150391, 0.23914474, 0.27173158, 0.31958289],
    [0.32641025, 0.26698576, 0.24129197, 0.25094166, 0.2501908,  0.25094166, 0.24129197, 0.26698576, 0.32641025],
    [0.29057492, 0.21326359, 0.15586953, 0.1693413,  0.19980031, 0.1693413, 0.15586953, 0.21326359, 0.29057492],
    [0.21951724, 0.12096531, 0.03968045, 0.01119779, 0.01379037, 0.01119779, 0.03968045, 0.12096531, 0.21951724]])

# construct training data in a form smt can use
N = 9
xta = np.linspace(-100,100,N)
xtb = np.linspace(-100,100,N)
xt = np.zeros((N*N,2))
index = 0
for i in range(N):
    for j in range(N):
        xt[index,:] = [xta[i],xtb[j]]
        index += 1

yt_ct = np.zeros((N*N,1))
index = 0
for i in range(N):
    for j in range(N):
        yt_ct[index,:] = ctarr[i,j]
        index += 1

yt_cp = np.zeros((N*N,1))
index = 0
for i in range(N):
    for j in range(N):
        yt_cp[index,:] = cparr[i,j]
        index += 1

xlimits = np.array([[-100.0, 100.0], [-100.0, 100.0]])

# construct surrogate model
sm_ct = RMTB(
            xlimits=xlimits,
            order=4,
            num_ctrl_pts=4,
            energy_weight=1e-10,
            regularization_weight=0.0,
            print_global=False,
            print_solver=False,)

sm_cp = RMTB(
            xlimits=xlimits,
            order=3,
            num_ctrl_pts=3, # 4 cp's and order 4 introduces local minimum
            energy_weight=1e-10,
            regularization_weight=0.0,
            print_global=False,
            print_solver=False,)

# train model
sm_ct.set_training_values(xt, yt_ct)
sm_ct.train()

sm_cp.set_training_values(xt, yt_cp)
sm_cp.train()


if __name__ == '__main__':
    # interpolate surrogate model and plot it
    num = 100
    x = np.linspace(-100,100,num)
    y = np.linspace(-100,100,num)
    X, Y = np.meshgrid(x, y)

    xint = np.zeros((num*num,2))
    index = 0
    for i in range(num):
        for j in range(num):
            xint[index,:] = [x[i],y[j]]
            index += 1

    zint_ct = sm_ct.predict_values(xint)
    ZCT = np.zeros((num,num))
    index = 0
    for i in range(num):
        for j in range(num):
            ZCT[i,j] = zint_ct[index]
            index += 1

    zint_cp = sm_cp.predict_values(xint)
    ZCP = np.zeros((num,num))
    index = 0
    for i in range(num):
        for j in range(num):
            ZCP[i,j] = zint_cp[index]
            index += 1



    plt.rcParams['figure.figsize'] = [14, 4]
    fontsize = 16


    levelsct = np.arange(0.15, 0.28, 0.005)
    levelscp = np.arange(0.13, 0.26, 0.005)
    fig, ((ax1), (ax2)) = plt.subplots(1, 2)

    

    plot_ct = ax1.contourf(X, Y, ZCT, cmap='plasma', levels=levelsct)
    plot_cp = ax2.contourf(X, Y, ZCP, cmap='viridis', levels=levelscp)

    ax1.contour(X,Y,ZCT,levels=levelsct,colors='k',alpha=0.2,linewidths=0.5)
    ax2.contour(X,Y,ZCP,levels=levelscp,colors='k',alpha=0.2,linewidths=0.5)

    xta = np.linspace(-100,100,15)
    xtb = np.linspace(-100,100,15)
    XX, YY = np.meshgrid(xta, xtb)
    ax1.scatter(XX, YY, marker='o', s=40, c='white', alpha=0.5, edgecolor='black')
    ax2.scatter(XX, YY, marker='o', s=40, c='white', alpha=0.5, edgecolor='black')

    cbar1 = plt.colorbar(plot_ct, shrink=1, ax=ax1,)
    cbar2 = plt.colorbar(plot_cp, shrink=1, ax=ax2)
    cbar1.ax.tick_params(labelsize=fontsize - 3)
    cbar2.ax.tick_params(labelsize=fontsize - 3)

    ax1.set_title('$C_T$', fontsize=fontsize)
    ax2.set_title('$C_P$', fontsize=fontsize)
    ax1.set_ylabel('axial inflow velocity (m/s)', fontsize=fontsize)
    ax2.set_ylabel('axial inflow velocity (m/s)', fontsize=fontsize)
    ax1.set_xlabel('edgewise inflow velocity (m/s)', fontsize=fontsize)
    ax2.set_xlabel('edgewise inflow velocity (m/s)', fontsize=fontsize)
    
    
    ax1.set_xlim(0,60)
    ax1.set_ylim(0,60)
    ax2.set_xlim(0,60)
    ax2.set_ylim(0,60)

    ax1.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize - 2)


    plt.savefig('rotor_model.pdf', transparent=True, bbox_inches='tight')

    # test predict vals
    #tt = np.array([[100,0]])
    #val = sm_ct.predict_values(tt)
    #print(val)

    plt.show()
    
    #plot = plt.contourf(xta, xtb, ctarr,cmap='plasma',levels=levelsct)