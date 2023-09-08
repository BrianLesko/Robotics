# Animation code in python 

# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Increases the plotted resolution
#plt.rcParams['figure.dpi'] = 300
#plt.rcParams['savefig.dpi'] = 300
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

# Create a meshgrid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
# Calculate Z
Z = np.sin(X) * np.cos(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')

for i in range(360):
    ax.clear()
    
    # set the labels and title of the plot 
    ax.set_xlabel('x - axis')
    ax.set_ylabel('y - axis')
    ax.set_zlabel('z - axis')
    ax.set_title('3D Line Plot')

    # replot the surface with updated view angle
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')

    # set the rotation of the 3d plot to be isometric
    ax.view_init(i, i)
    plt.pause(0.001)

plt.show()