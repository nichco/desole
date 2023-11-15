import matplotlib as mpl
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import numpy as np
import imageio

plt.rcParams["font.family"] = "Times New Roman"


path, attributes = svg2paths('gif/black4.svg')
marker = parse_path(attributes[0]['d'])
marker.vertices -= marker.vertices.mean(axis=0)
marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))


def create_frame(x, y, a, i, s, figsize, xlim, ylim, xlabel, ylabel, title, fontsize, marker_color):
    print('frame '+str(i))
    
    fig = plt.figure(figsize=figsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + a))

    # plt.axhline(y=300, color='black', linestyle='dashed', linewidth=1.5, alpha=0.5, zorder=1)
    # plt.fill_between(x_p, -100, obs_p, alpha=0.5, color='whitesmoke', hatch='//', edgecolor='lightgray', zorder=-1)
    # plt.plot(x_p, obs_p, c='black', alpha=0.5, linewidth=1, zorder=12, label='_nolegend_')
    # plt.legend(['target altitude', 'no-fly zone'], frameon=False, fontsize=fontsize - 2, loc='upper left')
    
    plt.plot(x[0:i], y[0:i], c=marker_color, alpha=0.4, linewidth=4, zorder=2, label='_nolegend_')
    plt.scatter(x[i], y[i], marker=new_marker, s=s, c=marker_color, zorder=3, alpha=1, label='_nolegend_')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    plt.savefig(f'img/img_{i}.png', transparent=True, dpi=150, facecolor='white', bbox_inches="tight")
    
    plt.close()


def combine_frames(time):
    frames = []
    for i, t in enumerate(time):
        image = imageio.v2.imread(f'img/img_{i}.png')
        frames.append(image)
        
    return frames


"""
# to save the gif with imageio
imageio.mimsave('./example.gif', # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second
"""





def interp_data(x, y, a, t, num):

    t_prime = np.arange(t[0], t[-1], t[-1]/num)

    x_prime = np.interp(t_prime, t, x)
    y_prime = np.interp(t_prime, t, y)
    a_prime = np.interp(t_prime, t, a)

    return x_prime, y_prime, a_prime, t_prime



