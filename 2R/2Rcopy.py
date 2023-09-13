##################################################################
# Brian Lesko 
# 9/7/2023
# Robotics Study, Forwawrd Kinematics, 2D Robot animation, Calculate the end effector position from the joint angles
# Linear algebra, screw axes approach
##################################################################

# The necessary library imports
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import modern_robotics as mr
from matplotlib import style
style.use("ggplot")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

# the home configuration of the robot is defined as the configuration where all the joint angles are zero
# It is most convenient to define the home configuration so that the rotational axis of each joint is aligned with the z-axis of the base frame, so that the math is less computationally expensive and easier to understand

# The link lengths are defined as L1 and L2, or L = [L1 L2]'
L1 = 1
L2 = 1

# The joint angles are defined as th1 and th2, or th = [th1 th2]'
# defining the joint angles as input variables in a streamlit app sidebar
th1 = st.sidebar.slider('Theta 1', -np.pi, np.pi, 0.0)
th2 = st.sidebar.slider('Theta 2', -np.pi, np.pi, 0.0)

# The home configuration of a 2R planar robot
p = np.array([[L1+L2], [0], [0]]) # The end effector position in the home configuration
M02 = np.block([[np.eye(3), p], [0, 0, 0, 1]]) # The end effector frame in the home configuration
p = np.array([[L1], [0], [0]]) # The first joint position in the home configuration
M01 = np.block([[np.eye(3), p], [0, 0, 0, 1]]) # The first joint frame in the home configuration
p = np.array([[0], [0], [0]]) # The base frame in the home configuration
M00 = np.block([[np.eye(3), p], [0, 0, 0, 1]]) # The base frame in the home configuration

# Screw Axis
# A screw axis is a line about which a rigid body move and rotate it is defined by a unit vector s and a point q that lies on the line 
s1 = np.array([[0], [0], [1], [0], [0], [0]]) 
s2 = np.array([[0], [0], [1], [0], [-L1], [0]])
Slist = np.hstack((s1, s2))

th = np.array([th1, th2]) # The joint angles

# Forward Kinematics
T02 = mr.FKinSpace(M02, Slist, th)
T01 = mr.FKinSpace(M01, s1, [th[0]])
T00 = mr.FKinSpace(M00, s1, [th[0]])

T_list = [T00,T01, T02]

# Create a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111)

# set the axes limits
ax.set_xlim([-2,4])
ax.set_ylim([-2,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
# set the axes ticks to only show integers
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

st.title('Forward Kinematics Demo')

# subtitle
st.write('By Brian Lesko, 9/12/2023')

# Plot the links 
for i in range(len(T_list)):
    if i == 0:
        ax.plot([0, T_list[i][0,3]], [0, T_list[i][1,3]], 'r-')
    if i > 0:
        ax.plot([T_list[i-1][0,3], T_list[i][0,3]], [T_list[i-1][1,3], T_list[i][1,3]], 'r-')

# Plot the joints
ax.plot(0, 0, 0, 'ro')
for i in range(len(T_list)-1):
    # extract the origin of each frame
    T = T_list[i]
    print(T)
    [x, y, z] = T[0:3, 3]
    # plot the origin
    ax.plot(x, y, 'ro')

######################################################
# Draw axes 
######################################################
def adjust_color(color, tone):
    # Convert the color from a hex code to RGB values
    rgb = [int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    
    # Adjust the RGB values based on the tone value
    if tone > 0.5:
        rgb = [min(x + (tone - 0.5) * 2 * (1 - x), 1) for x in rgb]
    else:
        rgb = [x * (tone * 2) for x in rgb]
    
    return rgb

def draw_axis(ax, T, length=4, tone=0.5,alpha=1.0):
    colors = ['#ff0000', '#00ff00', '#0000ff']
    origin = np.array([0, 0, 0, 1])  # Homogeneous coordinate for the origin
    
    for i in range(3):
        color = colors[i]
        
        # Adjust the color based on the tone value
        color = adjust_color(color, tone)

        # Define the end point of each axis in homogeneous coordinates
        end_point = np.array([0, 0, 0, 1])
        end_point[i] = length

        # Transform the origin and the end point using the transformation matrix T
        transformed_origin = np.dot(T, origin)
        transformed_end = np.dot(T, end_point)

        ax.quiver(transformed_origin[0], transformed_origin[1],
                  transformed_end[0] - transformed_origin[0],
                  transformed_end[1] - transformed_origin[1],
                  color=color, linewidth=2,alpha = alpha)

        # Plot a sphere at the end of the arrow tip using scatter, edit the size
        ax.scatter(transformed_end[0], transformed_end[1], s=1, color=color,alpha = .01)

# add a checkbox to show the draw axes at the joints and end effector
if st.sidebar.checkbox('Show Transformation Matrix'):
    # use drawaxes to draw the axes at the origin of each frame
    for i in range(len(T_list)):
        draw_axis(ax, T_list[i], length=1, tone=0.5, alpha=.2)
        # update the st plot

st.pyplot(fig)