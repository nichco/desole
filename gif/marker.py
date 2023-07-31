import matplotlib as mpl
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svgpath2mpl import parse_path

path, attributes = svg2paths('black4.svg')

new_marker = parse_path(attributes[0]['d'])

new_marker.vertices -= new_marker.vertices.mean(axis=0)

new_marker = new_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
new_marker = new_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))

x = 1
y = 1
#plt.plot(x,y,'o',marker=new_marker,markersize=0.1)


plt.scatter(1,1,marker=new_marker,s=30000)