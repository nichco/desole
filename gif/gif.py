import matplotlib as mpl
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import numpy as np
import imageio




path, attributes = svg2paths('black4.svg')
marker = parse_path(attributes[0]['d'])
marker.vertices -= marker.vertices.mean(axis=0)
marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))


def create_frame(x, y, a, t, s, figsize, xlim, ylim, xlabel, ylabel, title, fontsize):
    
    fig = plt.figure(figsize=figsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    new_marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(a))
    
    plt.scatter(x, y, marker=new_marker, s=s)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    
    plt.savefig(f'./img/img_{t}.png', transparent=False,  facecolor='white')
    
    plt.close()


def combine_frames(time):
    frames = []
    for t in time:
        image = imageio.v2.imread(f'./img/img_{t}.png')
        frames.append(image)
        
    return frames




#imageio.mimsave('./example.gif', # output gif
#                frames,          # array of input frames
#                fps = 5)         # optional: frames per second







