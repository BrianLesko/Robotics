##################################################################
# Brian Lesko 
# 9/21/2023
# Robotics Study, Forwawrd Dynamics, 2D Robot animation, Simulate a robot under given joint torques
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

#######################################################
# Forward Kinematics Functions, using the Screw Axes Approach
#######################################################
def getT_list(th):
    # The link lengths are defined as L1 and L2, or L = [L1 L2]'
    L1 = 1
    L2 = 1

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

    # Forward Kinematics
    T02 = mr.FKinSpace(M02, Slist, th)
    T01 = mr.FKinSpace(M01, s1, [th[0]])
    T00 = mr.FKinSpace(M00, s1, [th[0]])

    T_list = [T00,T01, T02]

    return T_list

######################################################
# Plot Links and Joints Functions
######################################################
def plot_robot(ax, T_list,alpha = 1.0):
    # Plot the links 
    for i in range(len(T_list)):
        if i == 0:
            ax.plot([0, T_list[i][0,3]], [0, T_list[i][1,3]], 'r-',alpha = alpha)
        if i > 0:
            ax.plot([T_list[i-1][0,3], T_list[i][0,3]], [T_list[i-1][1,3], T_list[i][1,3]], 'r-',alpha = alpha)

    # Plot the joints
    ax.plot(0, 0, 0, 'ro',alpha = alpha)
    for i in range(len(T_list)-1):
        # extract the origin of each frame
        T = T_list[i]
        print(T)
        [x, y, z] = T[0:3, 3]
        # plot the origin
        ax.plot(x, y, 'ro',alpha = alpha)

######################################################
# Draw axes Functions
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

def setPlotSettings(Joints = False):
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    # set the axes ticks to only show integers
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-2, 2])

    if Joints == True:
        for i in range(len(T_list)):
            draw_axis(ax1, T_list[i], length=1, tone=0.5, alpha=.333)

######################################################
# Gradient Descent functions 
# Motivational sources: Lecture 6, ECE 5307, Optimization methods
######################################################

def mcgeval(th,th_d):

    th = np.array(th)
    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2

    # The mass matrix
    M = 

    # The Coriolis matrix
    C = 

    # The gravity matrix
    G = 

    return M, C, G


def run(Jeval,th_init,th_d_init,lr=.01,n_iter=1000):

    # Initialize the joint parameters
    th = th_init
    th_d = th_d_init

    # Create history dictionary
    hist = {'th': [], 'th_d': [], 'th_dd': []}

    # Loop over iterations
    for i in range(n_iter):

        # Evaluate the equations of motion
        M, C, G = Jeval(th,th_d)

        Minv = np.linalg.inv(M)

        # Solve for th_dd using the equations of motion
        th_dd = Minv*(T - C - G) 

        # Save the history
        hist['th'].append(th)
        hist['th_d'].append(th_d)
        hist['th_dd'].append(th_dd)

        # Update the joint parameters
        th = th + lr * th_d
        th_d = th_d + lr * th_dd

    return hist

#######################################################
# TITLE and MAIN FIGURE
#######################################################
st.title('Robotics : Forward Dynamics')
st.write('By Brian Lesko, 9/21/2023')
"---"

# Select torques
col1, col2 = st.columns([5, 11], gap="medium")
with col1:
    st.write('This is a robot simulation ')
    st.write('Change its joint torques below 👇')
    T1 = st.slider('T1', -2.0, 2.0, 1.5, step=0.01)
    T2 = st.slider('T2', -2.0, 2.0, 0.5, step=0.01)
    T = np.array([[T1], [T2]])
    st.write('Check out the joint frames here 👇')
    Joints = False
    if st.checkbox('Show robot joints', value = False):
        Joints = True


# Additional options
subcol1, subcol2, subcol3 = st.columns(3)
with subcol1:
    Explanation = False
    if st.checkbox('Show Explanation Below'):
        Explanation = True
with subcol3:
    if st.checkbox('Change Initial Pose'):
        th1 = st.sidebar.slider('Theta 1', -np.pi, np.pi, 0.0, step=0.01)
        th2 = st.sidebar.slider('Theta 2', -np.pi, np.pi, 0.0, step=0.01)
        th0 = np.array([[th1], [th2]])
    else:
        th1 = 0
        th2 = 0
        th0 = np.array([[th1], [th2]])


######################################################
# basic figure
######################################################

# figure 1 is for the robot
fig1 = plt.figure(figsize=plt.figaspect(1))
ax1 = fig1.add_subplot(111)

# Plot the starting position
T_list = getT_list(th0)
plot_robot(ax1, T_list)
# Plot the goal position bold
ax1.plot(1.5, .5, 'kx', markersize=5)

######################################################
# PLACE the FIGURE in EXPANDER
######################################################

with col2:
    with st.expander("Dynamic 🤖", expanded=True):
            robot_plot = st.empty()

######################################################
# RUN the SIMULATIN
######################################################

hist = run(mcgeval,th0,th0,lr=.01,n_iter=1000)

######################################################
# ANIMATE the SIMULATION
######################################################

for i in range(len(hist['th'])):
    # Skip the iteration if the new joint parameters are < 1 degree from the previous iteration
    th = hist['th'][i]
    if i > 0:
        th_prev = hist['th'][i-1]
        if np.linalg.norm(th - th_prev) > .04 * np.pi/180:

            ax1.cla()
            T_list = getT_list(hist['th'][i])
            plot_robot(ax1, T_list,alpha = .333)
            ax1.plot(1.5, .5, 'kx', markersize=5)
            T_list = getT_list(hist['th'][i])
            plot_robot(ax1, T_list)
            setPlotSettings(Joints)

            with robot_plot:
                st.pyplot(fig1)

"---"
                 
#######################################################
# Explanation
#######################################################
