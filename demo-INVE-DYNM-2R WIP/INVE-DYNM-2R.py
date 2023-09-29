##################################################################
# Brian Lesko 
# 9/25/2023
# Robotics Study, Inverse Dynamics, 2D Robot animation, Simulate a robot under given joint torques
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

    # reshape theta to be (2,)
    th = np.reshape(th, (2,))

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
# Iterative Simulation Function
# Brian Lesko 9/21/23
######################################################

def mcgeval(th,th_d):
    
    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2
    r1 = .5 # Distance from joint 1 to center of mass of link 1
    r2 = .75 # Distance from joint 2 to center of mass of link 2
    m1 = .25  # Mass of link 1
    m2 = .25  # Mass of link 2
    g = 9.8  # Gravity

    # Mass Matrix
    M = np.array([
        [m1*L1**2 + m2*(L1**2 + 2*L1*r2*np.cos(th[1,0]) + L2**2), m2*(L1*r2*np.cos(th[1,0]) + L2**2)],
        [m2*(L1*r2*np.cos(th[1,0]) + r2**2), m2*r2**2]
    ])

    # Coriolis Vector
    C = np.array([
        [-m2*L1*r2*np.sin(th[1,0])*(2*th_d[0,0]*th_d[1,0] + th_d[1,0]**2)],
        [m2*L1*r2*th_d[0,0]**2*np.sin(th[1,0])]
    ])

    # Gravity Vector
    G = np.array([
        [(m1 + m2)*L1*g*np.cos(th[0,0]) + m2*g*r2*np.cos(th[0,0] + th[1,0])],
        [m2*g*r2*np.cos(th[0,0] + th[1,0])]
    ])

    # Center of Mass Positions
    x1 = r1 * np.cos(th[0,0])
    y1 = r1 * np.sin(th[0,0])
    com1 = np.array([[x1], [y1]])

    x2 = L1 * np.cos(th[0,0]) + r2 * np.cos(th[0,0] + th[1,0])
    y2 = L1 * np.sin(th[0,0]) + r2 * np.sin(th[0,0] + th[1,0])
    com2 = np.array([[x2], [y2]])

    return M, C, G, com1, com2

def run(Jeval,th_init,th_d_init,T,lr=.01,n_iter=1000):

    th = th_init
    th_d = th_d_init

    # Create history dictionary 
    hist = {'th': [], 'th_d': [], 'th_dd': [], 'M': [], 'C': [], 'G': [], 'T': [], 'X': [], 'com1': [], 'com2': [] }

    # Loop over iterations
    for i in range(n_iter):

        integrations = 25
        for i in range(integrations):

            # Evaluate the equations of motion
            M, C, G, com1,com2 = Jeval(th,th_d)

            # Frictional joint torques
            Friction = True
            if Friction == True:
                T_f = np.array([[0.2*th_d[0,0]], [0.2*th_d[1,0]]])

            # Solve for th_dd using the equations of motion
            th_dd = np.linalg.solve(M, T-T_f - C - G)

            # Update the joint parameters, Euler integration
            th_d = th_d + lr * th_dd
            th = th + lr * th_d

        # The end effector position 
        T_list = getT_list(th)
        Tend = T_list[2]
        X = Tend[0:3,3]

        # Save the history
        hist['th'].append(th)
        hist['th_d'].append(th_d)
        hist['th_dd'].append(th_dd)
        hist['M'].append(M)
        hist['C'].append(C)
        hist['G'].append(G)
        hist['T'].append(T)
        hist['X'].append(X)
        hist['com1'].append(com1)
        hist['com2'].append(com2)

    return hist

######################################################
# Inverse Dynamics Functions
#######################################################

# inputs desired theta, theta_d, theta_dd, and returns the joint torques
def inverseDynamics(th,th_d,th_dd):

    # Evaluate the equations of motion
    M, C, G, com1,com2 = mcgeval(th,th_d)

    Torques = np.dot(M,th_dd) + C + G 

    return Torques

#######################################################
# TITLE and MAIN FIGURE
#######################################################
st.title('Inverse Dynamics')
st.write('By Brian Lesko, 9/25/2023')

"---"

# Select torques
col1, col2 = st.columns([5, 11], gap="medium")
with col1:
    st.write('This is a robot simulation ')
    st.write('Change the desired joint angles 👇')
    th1 = st.number_input('Joint angle 1', -np.pi, np.pi, 0.0, step = 0.0001)
    th2 = st.number_input('Joint angle 2', -np.pi, np.pi, 0.0, step = 0.0001)
    th = np.array([[th1], [th2]])

    st.write('See the joint & link frames 👇')

# Additional options
subcol1, subcol2 = col1.columns([1,1], gap="medium")
with subcol1:
    Joints = False
    if st.checkbox('Joints'):
        Joints = True

with subcol2:
    # show center of mass
    Masses = False
    if st.checkbox('Masses'):
        Masses = True


st.write('  ')
st.write('**What is going on?**')
st.write('First, the program takes your desired joint angles and velocities and uses the equations of motion to calculate the needed torques; this is called inverse dynamics. Then, the the robot is simulated using a step input to these values. At each time step, the program takes the joint torques, positions, and velocities, and calculates the joint accelerations. The accelerations are then used to update the joint velocities and positions for the next iteration. This process is called forward dynamics in robotics engineering and is part of chapter 8 in the book Modern Robotics.')
st.write('  ')

suba, subb, subc = st.columns([1,1,1], gap="medium")

with suba:
    if st.checkbox('Change Initial Pose'):
        th1o = st.sidebar.number_input('Theta 1', -np.pi, np.pi, 0.0, step = 0.0001)
        th2o = st.sidebar.number_input('Theta 2', -np.pi, np.pi, 0.0, step = 0.0001)
        tho = np.array([[th1], [th2]])
        st.sidebar.warning('Start the robot from higher up to see double pendulum chaos behavior... To start from near rest set angles to -1.5708 and 0', icon="⚠️")
    else:
        th1o = -1.578
        th2o = 0
        tho = np.array([[th1o], [th2o]])

# which are derived from the Lagrangian of the system. The Lagrangian is the difference between the kinetic and potential energy of the system. The equations of motion are a set of 2nd order differential equations that can be solved for the joint accelerations. The equations of motion are solved using the numpy.linalg.solve function. The equations of motion are solved 20 times per iteration to improve accuracy.**')
st.write('  ')

######################################################
# basic figure
######################################################

# figure 1 is for the robot
fig1 = plt.figure(figsize=plt.figaspect(1))
ax1 = fig1.add_subplot(111)

# Plot the starting position
T_list = getT_list(th)
plot_robot(ax1, T_list)
# Plot the goal position bold
ax1.plot(1.5, .5, 'kx', markersize=5)

######################################################
# PLACE the FIGURE in EXPANDER
######################################################

with col2:
    with st.expander("Can you make it reach the X?", expanded=True):
        robot_plot = st.empty()


            ######################################################
            # RUN the SIMULATION
            ######################################################

        with st.spinner('Performing some simulation wizardry...'):
            th_d = np.array([[0], [0]])  # joint velocities\
            th_dd = np.array([[0], [0]])  # joint accelerations 
            T = inverseDynamics(th,th_d,th_dd)
            st.write(T)
            lr = .00009  # learning rate
            hist = run(mcgeval,tho,tho,T,lr,n_iter=3000)
            # write the shape of hist

######################################################
# Plot Angles over time 
######################################################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
total_time = len(hist['th'])*lr*20
ax2.plot(np.arange(0,total_time,lr*20),np.array(hist['th'])[:,0,0],alpha = 1)
ax2.plot(np.arange(0,total_time,lr*20),np.array(hist['th'])[:,1,0],alpha = 1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Joint Angle (rad)')
ax2.legend(['Joint 1','Joint 2'])
ax2.set_ylim(-2*np.pi,2*np.pi)

# Joint velocities
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(np.arange(0,total_time,lr*20),np.array(hist['th_d'])[:,0,0],alpha = 1)
ax3.plot(np.arange(0,total_time,lr*20),np.array(hist['th_d'])[:,1,0],alpha = 1)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Joint Velocity (rad/s)')
ax3.legend(['Joint 1','Joint 2'])
ax3.set_ylim(-2*np.pi,2*np.pi)

# Joint accelerations
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(np.arange(0,total_time,lr*20),np.array(hist['th_dd'])[:,0,0],alpha = 1)   
ax4.plot(np.arange(0,total_time,lr*20),np.array(hist['th_dd'])[:,1,0],alpha = 1)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Joint Acceleration (rad/s^2)')
ax4.legend(['Joint 1','Joint 2'])
ax4.set_ylim(-2*np.pi,2*np.pi)
ax4.tick_params(axis='both', which='major', labelsize=15)

with st.expander("Joint Angles", expanded=True):
    cubcol1, cubcol2, cubcol3 = st.columns([1,1,1], gap="medium")
    with cubcol1:
        st.pyplot(fig2)
    with cubcol2:
        st.pyplot(fig3)
    with cubcol3:
        st.pyplot(fig4)

######################################################
# ANIMATE the SIMULATION
######################################################

# select every nth frame

seq = np.arange(0, len(hist['th']), 55)

# add first frame to the beginning of the sequence
seq = np.insert(seq, 0, 0)

for i in seq:
    # Skip the iteration if the new joint parameters are < 1 degree from the previous iteration
    th = hist['th'][i]
    if i > 0:
        th_prev = hist['th'][i-1]
        if np.linalg.norm(th - th_prev) > .005 * np.pi/180:
            ax1.cla()

            # trace the end effector path from t = 0 to t = i using hist X
            X = np.array(hist['X'][0:i])
            ax1.plot(X[:,0], X[:,1], linestyle=(0, (10, 5)), color='k', alpha=1)

            # Plot the robot
            T_list = getT_list(hist['th'][i])
            plot_robot(ax1, T_list,alpha = .333)
            ax1.plot(1.5, .5, 'kx', markersize=5)
            T_list = getT_list(hist['th'][i])
            plot_robot(ax1, T_list)
            setPlotSettings(Joints)

            # Show the center of mass as a black cirlce with a white plus in the middle
            if Masses == True:
                com1 = np.array(hist['com1'][i])
                com2 = np.array(hist['com2'][i])
                ax1.plot(com1[0], com1[1], 'ko',alpha = 1)
                ax1.plot(com2[0], com2[1], 'ko',alpha = 1)
                ax1.plot(com1[0], com1[1], 'w+',alpha = 1, markersize = 6)
                ax1.plot(com2[0], com2[1], 'w+',alpha = 1, markersize = 6)

            with robot_plot:
                st.pyplot(fig1)

            # if the end effector is within .1 of the goal position
            if np.linalg.norm(X[-1,:] - np.array([[1.5], [.5]])) < .1:
                st.balloons()
                st.toast("🎉 You did it! 🎉", "Congratulations! You reached the goal position!", icon="🎉")
"---"
                 
#######################################################
# Explanation
#######################################################

Explanation = True
if Explanation == True:
    st.header('Forward Dynamics and Equations of Motion')

    st.write("""
    **What is Dynamics?**
            
    Dynamics is the study of how systems change in response to forces and torques. In robotics engineering, forward dynamics involves simulating a robot's behavior based on given joint torques, velocities, and positions. 
    """)

    st.write('**Steps Involved for Analytical Forward Dynamics**')
    st.write("""
    1. **Kinematics**: Determine each link's center of mass position and velocity to inform energy calculations.
    2. **Kinetic Energy**: Assess kinetic energy for each link, factoring in joint velocities and link attributes.
    3. **Potential Energy**: Evaluate based on gravitational effects within the Lagrangian framework.
    4. **Lagrangian**: Opt for this approach over Newton Euler; it emphasizes the energy difference and excels with robots having multiple DOFs.
    5. **Equations of Motion**: Formulate these equations using the Lagrangian, applying differentiation techniques.
    6. **Matrix Form**: Simplify intricate motion equations into a matrix layout, aiding in simulation tasks.
    7. **Numerical Methods**: Given the intricacy of analytical methods for high DOF robots, numerical solutions are often more practical.
    8. **Non-Physics based methods**: With the rise of machine learning, new non-traditional approaches are emerging for robotic control. It's essential to grasp both these and conventional physics-driven methods.
    """)

    "---"

    st.subheader('The Mathematics Behind the Simulation')
    st.write('I havent gone into much detail explaining how these were derived, but I will give a brief overview of the math behind the simulation.')
    
    with st.expander("Kinematics", expanded=False):
        st.write('For Link 1')
        st.latex(r'''
        \begin{align}
        \begin{bmatrix}
        x_1 \\
        y_1 
        \end{bmatrix}
        &=
        \begin{bmatrix}
        r_1 \cos(\theta_1) \\
        r_1 \sin(\theta_1)
        \end{bmatrix}
        \end{align}
        ''')
        st.latex(r'''
        \begin{align}
        \begin{bmatrix}
        \dot{x}_1 \\
        \dot{y}_1
        \end{bmatrix}
        &=
        \begin{bmatrix}
        -r_1 \sin(\theta_1) \\
        r_1 \cos(\theta_1)
        \end{bmatrix} \dot{\theta_1}
        \end{align}
        ''')

        st.write('For Link 2')
        st.latex(r'''
        \begin{align}
        \begin{bmatrix}
        x_2 \\
        y_2 
        \end{bmatrix}
        &=
        \begin{bmatrix}
        L_1 \cos(\theta_1) + r_2 \cos(\theta_1 + \theta_2) \\
        L_1 \sin(\theta_1) + r_2 \sin(\theta_1 + \theta_2)
        \end{bmatrix}
        \end{align}
        ''')
        st.latex(r'''
        \begin{align}
        \begin{bmatrix}
        \dot{x}_2 \\
        \dot{y}_2
        \end{bmatrix}
        &=
        \begin{bmatrix}
        -L_1 \sin(\theta_1) - r_2 \sin(\theta_1 + \theta_2) & - r_2 \sin(\theta_1 + \theta_2) \\
        L_1 \cos(\theta_1) + r_2 \cos(\theta_1 + \theta_2) & r_2 \cos(\theta_1 + \theta_2)
        \end{bmatrix}
        \begin{bmatrix}
        \dot{\theta_1} \\
        \dot{\theta_2}
        \end{bmatrix}
        \end{align}
        ''')
        st.write('')
        st.write('')

    with st.expander("Kinetic Energy", expanded=False):
        st.write('For Link 1')
        st.latex(r'''
        \begin{equation}
        \begin{aligned}
        K_1 = \frac{1}{2} m_1 \bigg[ &\dot{x}_1^2 + \dot{y}_1^2 \\
        &+ (-r_1 \sin(\theta_1) \dot{\theta_1})^2 + (r_1 \cos(\theta_1) \dot{\theta_1})^2 \\
        &+ r_1^2 \dot{\theta_1}^2 \bigg]
        \end{aligned}
        \end{equation}
        ''')
        st.write('For Link 2')
        st.latex(r'''
        \begin{equation}
        \begin{aligned}
        K_2 = \frac{1}{2} M_2 \bigg[ &(L_1^2 + 2L_1 r_2 \cos(\theta_2) + r_2^2) \dot{\theta_1}^2 \\
        &+ 2 (L_1 r_2 \cos(\theta_2) + r_2^2) \dot{\theta_1} \dot{\theta_2} \\
        &+ r_2^2 \dot{\theta_2}^2 \bigg]
        \end{aligned}
        \end{equation}
        ''')
        st.write('')
        st.write('')
    with st.expander("Potential Energy", expanded=False):
        st.write('For Link 1')
        st.latex(r'''
        \begin{align}
        PE_1 &= g \cdot m_1 \cdot r_1 \cdot \cos(\theta_1)
        \end{align}
        ''')
        st.write('For Link 2')
        st.latex(r'''
        \begin{align}
        PE_2 &= g \cdot m_2 \left( L_1 \cos(\theta_1) + r_2 \cos(\theta_1 + \theta_2) \right)
        \end{align}
        ''')
        st.write('')
        st.write('')
    with st.expander("Lagrangian", expanded=False):
        st.latex(r'''
        \begin{equation}
        \mathcal{L} = K - P
        \end{equation}
        ''')

        st.latex(r'''
        \begin{align}
        \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}_i} \right) - \frac{\partial \mathcal{L}}{\partial \theta_i} &= \tau_i
        \end{align}
        ''')
        st.write('')
        st.write('')
    with st.expander("Equations of Motion", expanded=False):
        st.latex(r'''
        \begin{equation}
        \begin{aligned}
        \tau_1 &= \left( m_1 L_1^2 + m_2 \left( L_1^2 + 2L_1 r_2 \cos(\theta_2) + L_2^2 \right) \right) \ddot{\theta_1} \\
        &+ m_2 \left( L_1 r_2 \cos(\theta_2) + L_2^2 \right) \ddot{\theta_2} \\
        &- m_2 L_1 r_2 \sin(\theta_2) \left( 2\dot{\theta_1} \dot{\theta_2} + \dot{\theta_2}^2 \right) \\
        &+ (m_1 + m_2) L_1 g \cos(\theta_1) + m_2 g r_2 \cos(\theta_1 + \theta_2) \\
        \tau_2 &= m_2 \left( L_1 r_2 \cos(\theta_2) + r_2^2 \right) \ddot{\theta_1} \\
        &+ m_2 r_2^2 \ddot{\theta_2} \\
        &+ m_2 L_1 r_2 \dot{\theta_1}^2 \sin(\theta_2) \\
        &+ m_2 g r_2 \cos(\theta_1 + \theta_2)
        \end{aligned}
        \end{equation}
        ''')
        st.write('')
        st.write('')
    with st.expander("Matrix Form", expanded=False):
        st.latex(r'''
        \begin{equation}
        \mathbf{M}(\theta) \cdot \ddot{\theta} + \mathbf{C}(\theta, \dot{\theta}) + \mathbf{G}(\theta) = \boldsymbol{\tau}
        \end{equation}
        ''')

        st.latex(r'''
        \begin{equation}
        \mathbf{M}(\theta) = \begin{bmatrix}
        m_1 L_1^2 + m_2 \left( L_1^2 + 2L_1r_2\cos(\theta_2) + L_2^2 \right) & m_2 \left( L_1 r_2 \cos(\theta_2) + L_2^2 \right) \\
        m_2 \left( L_1 r_2 \cos(\theta_2) + r_2^2 \right) & m_2 r_2^2
        \end{bmatrix}
        \end{equation}
        ''')

        st.latex(r'''
        \begin{equation}
        \mathbf{C}(\theta, \dot{\theta}) = \begin{bmatrix}
        -m_2 L_1 r_2 \sin(\theta_2) \left(2\dot{\theta}_1 \dot{\theta}_2 + \dot{\theta}_2^2 \right) \\
        m_2 L_1 r_2 \dot{\theta}_1^2 \sin(\theta_2)
        \end{bmatrix}
        \end{equation}
        ''')

        st.latex(r'''
        \begin{equation}
        \mathbf{G}(\theta) = \begin{bmatrix}
        (m_1 + m_2) L_1 g \cos(\theta_1) + m_2 g r_2 \cos(\theta_1 + \theta_2) \\
        m_2 g r_2 \cos(\theta_1 + \theta_2)
        \end{bmatrix}
        \end{equation}
        ''')

        st.latex(r'''
        \begin{equation}
        \boldsymbol{\tau} = \begin{bmatrix} \tau_1 \\ \tau_2 \end{bmatrix}
        \end{equation}
        ''')
        st.write('')
        st.write('')

"---"
st.subheader('The Code Behind the Simulation')

with st.expander("The Code", expanded=True):
    st.code("""
    def mcgeval(th,th_d):
    
    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2
    r1 = .5 # Distance from joint 1 to center of mass of link 1
    r2 = .75 # Distance from joint 2 to center of mass of link 2
    m1 = .25  # Mass of link 1
    m2 = .25  # Mass of link 2
    g = 9.8  # Gravity

    # Mass Matrix
    M = np.array([
        [m1*L1**2 + m2*(L1**2 + 2*L1*r2*np.cos(th[1,0]) + L2**2), m2*(L1*r2*np.cos(th[1,0]) + L2**2)],
        [m2*(L1*r2*np.cos(th[1,0]) + r2**2), m2*r2**2]
    ])

    # Coriolis Vector
    C = np.array([
        [-m2*L1*r2*np.sin(th[1,0])*(2*th_d[0,0]*th_d[1,0] + th_d[1,0]**2)],
        [m2*L1*r2*th_d[0,0]**2*np.sin(th[1,0])]
    ])

    # Gravity Vector
    G = np.array([
        [(m1 + m2)*L1*g*np.cos(th[0,0]) + m2*g*r2*np.cos(th[0,0] + th[1,0])],
        [m2*g*r2*np.cos(th[0,0] + th[1,0])]
    ])

    # Center of Mass Positions
    x1 = r1 * np.cos(th[0,0])
    y1 = r1 * np.sin(th[0,0])
    com1 = np.array([[x1], [y1]])

    x2 = L1 * np.cos(th[0,0]) + r2 * np.cos(th[0,0] + th[1,0])
    y2 = L1 * np.sin(th[0,0]) + r2 * np.sin(th[0,0] + th[1,0])
    com2 = np.array([[x2], [y2]])

    return M, C, G, com1, com2

def run(Jeval,th_init,th_d_init,T,lr=.01,n_iter=1000):

    th = th_init
    th_d = th_d_init

    # Create history dictionary 
    hist = {'th': [], 'th_d': [], 'th_dd': [], 'M': [], 'C': [], 'G': [], 'T': [], 'X': [], 'com1': [], 'com2': [] }

    # Loop over iterations
    for i in range(n_iter):

        integrations = 20
        for i in range(integrations):

            # Evaluate the equations of motion
            M, C, G, com1,com2 = Jeval(th,th_d)

            # Frictional joint torques
            Friction = True
            if Friction == True:
                T_f = np.array([[0.2*th_d[0,0]], [0.2*th_d[1,0]]])

            # Solve for th_dd using the equations of motion
            th_dd = np.linalg.solve(M, T-T_f - C - G)

            # Update the joint parameters, Euler integration
            th_d = th_d + lr * th_dd
            th = th + lr * th_d

        # The end effector position 
        T_list = getT_list(th)
        Tend = T_list[2]
        X = Tend[0:3,3]

        # Save the history
        hist['th'].append(th)
        hist['th_d'].append(th_d)
        hist['th_dd'].append(th_dd)
        hist['M'].append(M)
        hist['C'].append(C)
        hist['G'].append(G)
        hist['T'].append(T)
        hist['X'].append(X)
        hist['com1'].append(com1)
        hist['com2'].append(com2)

    return hist
    """, language='python')

"---"
    
#######################################################
# Brian Lesko
# 9/21/2023
#######################################################

col1, col2, = st.columns([1,5], gap="medium")

with col1:
    st.image('./dp.png')

with col2:
    st.write(""" 
    Hey it's Brian,
            
    Thanks for visiting my forward dynamics page. I hope you find it brilliant!
            
    Feel free to connect with me on Twitter, Github, and LinkedIn.
    """)

##################################################################
# Brian Lesko
# Social Links
##################################################################

# Make 10 columns 
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col2:
    st.write('')
    st.write('[Twitter](https://twitter.com/BrianJosephLeko)')

with col3:
    st.write('')
    st.write('[LinkedIn](https://www.linkedin.com/in/brianlesko/)')

with col4:
    st.write('')
    st.write('[Github](https://github.com/BrianLesko)') 

with col5:
    st.write('')
    st.write('[Buy me a Coffee](https://www.buymeacoffee.com/brianlesko)')

"---"
