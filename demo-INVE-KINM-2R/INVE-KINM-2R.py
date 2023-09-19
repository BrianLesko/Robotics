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

#######################################################
# Forward Kinematics Functions, using the Screw Axes Approach
#######################################################
L1 = 1
L2 = 1

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
# Gradient Descent functions 
# Motivational sources: Lecture 6, ECE 5307, Optimization methods
######################################################

def Jeval(th, X_goal):
    # Constants (replace with actual values)
    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2

    # Make th a numpy array
    th = np.array(th)

    # Compute the end effector position
    x = L1*np.cos(th[0]) + L2*np.cos(th.sum())
    y = L1*np.sin(th[0]) + L2*np.sin(th.sum())
    X = np.array([[x], [y]])

    # Compute the cost function
    J = np.sqrt((x - X_goal[0])**2 + (y - X_goal[1])**2)

    # Compute the gradients
    J_grad_x = 2*(x - X_goal[0]) * (-(L1*np.sin(th[0])) - L2*np.sin(th.sum())) + 2*(y - X_goal[1]) * (L1*np.cos(th[0]) + L2*np.cos(th.sum()))
    J_grad_y = -2*L2*(x - X_goal[0])*np.sin(th.sum()) + 2*L2*(y - X_goal[1])*np.cos(th.sum())
    
    J_grad = np.array([J_grad_x, J_grad_y]) / (2*J)

    return J, J_grad


def grad_opt_simp(Jeval,th_init,X_goal,lr=1e-3,n_iter=1000):
    """
    Simple gradient descent optimization
    
    Jeval: function that evaluates the cost and gradient

    th_init: initial guess of the joint parameters
    lr: learning rate
    n_iter: number of iterations
    """
    # Initialize the joint parameters
    th0 = th_init

    # Create history dictionary
    hist = {'th0': [], 'J': []}

    # Loop over iterations
    for i in range(n_iter):

        # Evaluate the cost and gradient
        J0, J_grad0 = Jeval(th0,X_goal)

        # Save the history
        hist['th0'].append(th0)
        hist['J'].append(J0)

        # Update the joint parameters
        th0 = th0 - lr * J_grad0

    return th0, J0, hist

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


#######################################################
# MAKE COLUMNS, SIDEBAR, GET INPUTS, TITLE
#######################################################

st.title('Robotics : Inverse Kinematics')

# subtitle
st.write('By Brian Lesko, 9/13/2023')

"---"

col1, col2 = st.columns([5, 11], gap="medium")

with col1:

    right_side = col1.radio(
    "Show on right side 👉", ["Dynamic", "Static"], horizontal=True)

    x_goal = st.slider('X Goal', -2.0, 2.0, 1.5, step=0.01)
    y_goal = st.slider('Y Goal', -2.0, 2.0, 0.5, step=0.01)
    X_goal = np.array([[x_goal], [y_goal]])

    st.write('Cost Surface', '📈')
    cost_surface = st.empty()

st.write('  ')
subcol1, subcol2, subcol3 = st.columns(3)
with subcol1:
    Joints = False
    if st.checkbox('Show robot joints'):
        Joints = True
    
with subcol3:
    Explanation = False
    if st.checkbox('Show Explanation Below'):
        Explanation = True

with subcol2:
    if st.checkbox('Change Initial Pose'):
        th1 = st.sidebar.slider('Theta 1', -np.pi, np.pi, 0.0, step=0.01)
        th2 = st.sidebar.slider('Theta 2', -np.pi, np.pi, 0.0, step=0.01)
        th0 = np.array([[th1], [th2]])
    else:
        th1 = 0
        th2 = 0
        th0 = np.array([[th1], [th2]])

"---"

#######################################################
# INITIALIZATION OF THE FIRST PLOT AND TITLE
#######################################################

# figure 1 is for the robot
fig1 = plt.figure(figsize=plt.figaspect(0.75))
ax1 = fig1.add_subplot(111)

# figure 2 is for the cost surface
fig2 = plt.figure(figsize=plt.figaspect(0.75))
ax2 = fig2.add_subplot(111, projection='3d')

def setPlotSettings(Joints = False,azim = -60):
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    # set the axes ticks to only show integers
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # set text size for figure 2
    ax2.tick_params(labelsize=6)
    ax2.set_xlabel('Theta 1',fontsize=6)
    ax2.set_ylabel('Theta 2', fontsize=6)
    ax2.set_zlabel('Cost', fontsize=6)

    # Increase the figure size
    fig2.set_size_inches(4, 4)

    # decrease tick mark size for surface plot
    ax2.tick_params(labelsize=5)
    ax2.locator_params(axis='x', nbins=3)
    ax2.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='z', nbins=3)

    # set the elevation and azimuth of the surface plot
    ax2.view_init(elev=45, azim=azim)

    # decrease the space between the axes ticks and the plot 
    ax2.tick_params(axis='x', pad=-3)
    ax2.tick_params(axis='y', pad=-3)
    ax2.tick_params(axis='z', pad=-3)

    # decrease the space between the axes labels and the plot
    ax2.xaxis.labelpad = -10
    ax2.yaxis.labelpad = -10
    ax2.zaxis.labelpad = -10

    ax1.set_xlim([-2,4])
    ax1.set_ylim([-2,2])

    # set the axes label to theta_1 and theta_2 in latex
    ax2.set_xlabel(r'$\theta_1$', fontsize=6)
    ax2.set_ylabel(r'$\theta_2$', fontsize=6)
    ax2.set_zlabel('J', fontsize=6)

    if Joints == True:
        for i in range(len(T_list)):
            draw_axis(ax1, T_list[i], length=1, tone=0.5, alpha=.2)

######################################################
# PLOT the START and GOAL robot position
######################################################

# Plot the robot
T_list = getT_list(th0)
plot_robot(ax1, T_list)

# Plot the goal position bold
ax1.plot(X_goal[0], X_goal[1], 'kx', markersize=5)

######################################################
# Run the gradient descent optimization
######################################################

lr = 0.0045
th, J, hist = grad_opt_simp(Jeval,th0,X_goal,lr,n_iter=2000)

# plot the location of the minimum cost
ax2.scatter(th[0], th[1], J, c='k', marker='x', s=15)

# plot the location of the starting cost
ax2.scatter(th0[0], th0[1], hist['J'][0], c='k', marker='o', s=15)

# find the average of th and th0
th_avg = (th + th0) / 2
# plot the location of the average cost for debugg 
#ax2.scatter(th_avg[0], th_avg[1], Jeval(th_avg,X_goal)[0], c='k', marker='o', s=15)

T_list0 = getT_list(th0)
T_listEnd = getT_list(th)
plot_robot(ax1, T_listEnd,alpha = .1)

###################################################### 
# Plot the cost surface
######################################################

# create a meshgrid of theta values centered at th_avg
c = 1
th1 = np.linspace(-c*np.pi, c*np.pi, 18) + th_avg[0]
th2 = np.linspace(-c*np.pi, c*np.pi, 18) + th_avg[1]
TH1, TH2 = np.meshgrid(th1, th2)

# evaluate the cost function for each theta combination
J = np.zeros(TH1.shape)
for i in range(TH1.shape[0]):
    for j in range(TH1.shape[1]):
        J[i,j], _ = Jeval([TH1[i,j], TH2[i,j]],X_goal)

# plot the cost surface
ax2.plot_surface(TH1, TH2, J, cmap='plasma', linewidth=0.2, alpha=.6)
#ax2.plot_wireframe(TH1, TH2, J, rstride=1, cstride=1,color = 'red' ,alpha = 1)

# plot the history of the gradient descent optimization
ax2.plot(np.array(hist['th0'])[:,0], np.array(hist['th0'])[:,1], hist['J'], 'k-', linewidth=1.5,alpha = 1)

#######################################################
# One plot line reduces plot flickering
#######################################################

if right_side == "Dynamic":
    
    with col2:

        with st.expander("Dynamic 🦾", expanded=True):
            plotspot = st.empty()

    with col2:
        with st.expander("Optimization Information"):
            subcol1, subcol2 = st.columns(2)
            with subcol1: 
                st.write('Initial cost: ', hist['J'][0])
                        
            with subcol2:
                st.write('Final cost: ', hist['J'][-1])
                    
            st.write('Number of iterations: ', len(hist['th0']))
            st.write('Learning rate: ', lr)
            st.write('Final Error: ', np.linalg.norm(T_listEnd[2][0:2,3] - X_goal.T),'meters')
    
    with col2:
        with st.expander("Cost Surface Adjustment"):
            st.write('Empty for now')

            

    n = len(hist['th0'])
    # Generate exponential sequence
    exp_seq = np.geomspace(1, n, num=50, dtype=int) - 1

    azim = -60
    for i in exp_seq:

        # Skip the iteration if the new joint parameters are < 1 degree from the previous iteration
        th = hist['th0'][i]
        if i > 0:
            th_prev = hist['th0'][i-1]
            if np.linalg.norm(th - th_prev) > .04 * np.pi/180:
                # clear the axes
                ax1.clear()
                ax2.clear()
                # plot the goal position and starting position
                ax1.plot(X_goal[0], X_goal[1], 'kx', markersize=5)
                plot_robot(ax1, T_list0,alpha = .1)
                # plot the robot
                T_list = getT_list(hist['th0'][i])
                plot_robot(ax1, T_list)
                azim = azim + 2
                setPlotSettings(Joints,azim)
                # update the st plot
                plot_robot(ax1, T_list,alpha = .5)
                with plotspot: 
                    st.pyplot(fig1)
                # update the cost surface
                ax2.plot_surface(TH1, TH2, J, cmap='plasma', linewidth=0.2, alpha=.5)
                ax2.plot(np.array(hist['th0'])[:,0], np.array(hist['th0'])[:,1], hist['J'], 'k-', linewidth=1.5,alpha = 1)
                ax2.scatter(th[0], th[1], Jeval(th,X_goal)[0], c='red', marker='x', s=20)
                ax2.scatter(th0[0], th0[1], hist['J'][0], c='k', marker='o', s=15)

                with cost_surface:
                    st.pyplot(fig2)

if right_side == "Static":
    setPlotSettings(Joints)
    cost_surface.pyplot(fig2)
    with col2:

        with st.expander("Static 🦾", expanded=True):
            setPlotSettings(Joints)
            st.pyplot(fig1)
    with col2:
        with st.expander("Optimization Information"):
            subcol1, subcol2 = st.columns(2)
            with subcol1: 
                st.write('Initial cost: ', hist['J'][0])
                        
            with subcol2:
                st.write('Final cost: ', hist['J'][-1])
                    
            st.write('Number of iterations: ', len(hist['th0']))
            st.write('Learning rate: ', lr)
            st.write('Final Error: ', np.linalg.norm(T_listEnd[2][0:2,3] - X_goal.T),'meters')
    with col2:
        with st.expander("Cost Surface Adjustment"):
            st.write('Empty for now')
                 
#######################################################
# Explanation
#######################################################

#if explanation exists, show it
if Explanation == True:
    st.write('  ')

    st.title('Gradient Descent & Optimization')

    st.write("""

    **What is Gradient Descent?**

    Gradient descent is useful for robotics and AI. It's a bit like a mountain climber finding the best path down a hill.

    **How Does It Work?**

    1. **Cost Function**: Imagine the robot's goal is to reach a point, and we have a way to measure how far it is from that point. This measurement is called the cost function.
    
    2. **Gradient**: To help the robot move closer to the goal, we calculate the gradient. It's a compass that tells us the direction where the cost function decreases the fastest. 

    3. **Learning Rate**: This is a setting we use to control how big of a step the robot takes towards the goal in each move. A small step might take longer but is more accurate, while a big step is faster but might miss the best path.

    **The Process**

    We repeat the steps of calculating the gradient and moving the robot step by step until it reaches the goal. This whole process is what we call gradient descent optimization.

    **How to Represent it in Math**

    The mathematical formula for updating the robot's position during this process is:

    And that's how we use gradient descent to help robots move to a goal position!

    **In Python code**
    
    """)

    # include my jeval and grad_opt_simp functions
    body = """def Jeval(th, X_goal):
    # Constants (replace with actual values)
    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2

    # Make th a numpy array
    th = np.array(th)

    # Compute the end effector position
    x = L1*np.cos(th[0]) + L2*np.cos(th.sum())
    y = L1*np.sin(th[0]) + L2*np.sin(th.sum())
    X = np.array([[x], [y]])

    # Compute the cost function
    J = np.sqrt((x - X_goal[0])**2 + (y - X_goal[1])**2)

    # Compute the gradients
    J_grad_x = 2*(x - X_goal[0]) * (-(L1*np.sin(th[0])) - L2*np.sin(th.sum())) + 2*(y - X_goal[1]) * (L1*np.cos(th[0]) + L2*np.cos(th.sum()))
    J_grad_y = -2*L2*(x - X_goal[0])*np.sin(th.sum()) + 2*L2*(y - X_goal[1])*np.cos(th.sum())
    
    J_grad = np.array([J_grad_x, J_grad_y]) / (2*J)

    return J, J_grad


def grad_opt_simp(Jeval,th_init,X_goal,lr=1e-3,n_iter=1000):

    # Initialize the joint parameters
    th0 = th_init

    # Create history dictionary
    hist = {'th0': [], 'J': []}

    # Loop over iterations
    for i in range(n_iter):

        # Evaluate the cost and gradient
        J0, J_grad0 = Jeval(th0,X_goal)

        # Save the history
        hist['th0'].append(th0)
        hist['J'].append(J0)

        # Update the joint parameters
        th0 = th0 - lr * J_grad0

    return th0, J0, hist"""

    st.code(body, language="python", line_numbers=False)


    st.write('  ')
    st.write(' Find a more detailed explanation of this topic by Northwestern in [*Modern Robotics*](https://modernrobotics.northwestern.edu/nu-gm-book-resource/inverse-kinematics-of-open-chains/#department).')

    "---"
    
st.write('  ') 
st.write('  ') 

st.write(""" 
Hey it's Brian,
         
Through this page, I want to share my passion for engineering and my desire to be at the forefront of where technology meets creativity and precision.
         
Thanks for visting the page, I hope you enjoy it!
         
Feel free to explore more about my journey and connect with me through [Twitter](https://twitter.com/BrianJosephLeko), [Linkedin](https://www.linkedin.com/in/brianlesko/), or [Github](https://github.com/BrianLesko).

""")

"---"
