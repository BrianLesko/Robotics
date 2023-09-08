##################################################################
# Brian Lesko 
# 9/7/2023
# Robotics Study, Transformation Matricies, Axes animation
##################################################################

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

# increase matplotlib resolution
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
#import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300

# use times new roman font in matpotlib figures
plt.rcParams["font.family"] = "Times New Roman"

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

        ax.quiver(transformed_origin[0], transformed_origin[1], transformed_origin[2],
                  transformed_end[0] - transformed_origin[0],
                  transformed_end[1] - transformed_origin[1],
                  transformed_end[2] - transformed_origin[2],
                  color=color, linewidth=2, arrow_length_ratio=0.07,alpha = alpha)

        # Plot a sphere at the end of the arrow tip using scatter, edit the size
        ax.scatter(transformed_end[0], transformed_end[1], transformed_end[2], s=1, color=color,alpha = alpha)


def transformationMatrix(Tstart, x, y, z, xangle, yangle, zangle):
    # Precompute cosine and sine values
    cos_x, sin_x = np.cos(xangle), np.sin(xangle)
    cos_y, sin_y = np.cos(yangle), np.sin(yangle)
    cos_z, sin_z = np.cos(zangle), np.sin(zangle)
    
    # Compute individual rotation matrices using precomputed values
    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    
    # Combine rotation matrices
    R = Rx @ Ry @ Rz

    # Create and populate the transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]

    # Compute the new transformation matrix
    Tnew = Tstart @ T
    return Tnew



# use streamlit to create a web app to control the transformation matrix
import streamlit as st

st.title('Transformation Matricies Demo')

# subtitle
st.write('By Brian Lesko, 9/08/2023')

# create a widget for each transformation matrix parameter
x = st.sidebar.slider('Translation in the X direction', -2.0, 2.0, 0.0)
y = st.sidebar.slider('Translation in the Y direction', -2.0, 2.0, 0.0)
z = st.sidebar.slider('Translation in the Z direction', -2.0, 2.0, 0.0)
xangle = st.sidebar.slider('Rotation about the X axis', -np.pi, np.pi, 0.0)
yangle = st.sidebar.slider('Rotation about the Y axis', -np.pi, np.pi, 0.0)
zangle = st.sidebar.slider('Rotation about the Z axis', -np.pi, np.pi, -np.pi/2)

# Non-interactive elements return a placeholder to their location
# in the app. Here we're storing progress_bar to update it later.
progress_bar = st.sidebar.progress(0)

# These two elements will be filled in later, so we create a placeholder
# for them using st.empty()
frame_text = st.sidebar.empty()
plot_spot = st.empty()

def reset_axis(ax):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

# create a figure
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
reset_axis(ax)
fig.patch.set_facecolor('#E5E5E5')
plt.rcParams['axes.titleweight'] = 'bold'

Tlist = []
Tinit = np.eye(4)  # Initial transformation matrix

for frame_num, t in enumerate(np.linspace(0.0, 1, 10)):  # Adjusting the linspace range to 0 and 1
    # Here were setting value for these two elements.
    progress_bar.progress(frame_num*10)
    frame_text.text("Frame %i/10" % (frame_num + 1))

    # If Tlist is not empty, use the last element as Tlast, else use Tinit
    Tlast = Tlist[-1] if Tlist else Tinit

    # Performing some Linear Algebra wizardry.
    T = transformationMatrix(Tinit, x*t, y*t, z*t, xangle*t, yangle*t, zangle*t)

    # Adding the new transformation matrix to Tlist
    Tlist.append(T)

    # draw the coordinate frame
    draw_axis(ax, T, tone=.45)

    with plot_spot:
        st.pyplot(fig)
    ax.clear()
    reset_axis(ax)



# We clear elements by calling empty on them.
progress_bar.empty()
frame_text.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")


#################################################
# Mathematical Explanation
#################################################
if st.sidebar.checkbox("explanation", value=True):
    # explain what a rotation matrix is
    st.write('  ')
    st.write('  ')
    st.write('A rotation matrix is a 3x3 matrix that represents orientation using three vectors.')
    st.write('The first vector is the x axis, the second vector is the y axis, and the third vector is the z axis, each as a unit vector.')

    # The standard coordinate frame is the identity matrix, with the x axis being the first column, the y axis being the second column, and the z axis being the third column
    st.write('The standard coordinate frame is the identity matrix, with the x axis being the first column, the y axis being the second column, and the z axis being the third column')

    # A rotation matrix can be written using the following formula
    st.write('A rotation matrix can also be written using the following formula')

    # write the formula for each rotation matrix in latex to the streamlit app
    st.latex(r'''
    \begin{align}
    R_z &= \begin{bmatrix}
    \cos(\theta) & -\sin(\theta) & 0 \\
    \sin(\theta) & \cos(\theta) & 0 \\
    0 & 0 & 1
    \end{bmatrix} \\
    R_y &= \begin{bmatrix}
    \cos(\theta) & 0 & \sin(\theta) \\
    0 & 1 & 0 \\
    -\sin(\theta) & 0 & \cos(\theta)
    \end{bmatrix} \\
    R_x &= \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos(\theta) & -\sin(\theta) \\
    0 & \sin(\theta) & \cos(\theta)
    \end{bmatrix} \\
    \end{align}
    ''')

    # explain how to multiply rotation matricies         
    st.write('To rotate a coordinate frame about multiple axis, you multiply the rotation matricies together')
            
    # the formula for the Rotation matrix is the product of the three rotation matricies
    st.latex(r'''
    \begin{align}
    R &= R_x R_y R_z \\
    \end{align}
    ''')
        
    st.write('An interesting property of the rotation matrix is orthogonality, meaning that the inverse is equal to the transpose')

    # rotation matricies can rotate vectors
    st.write('Overall, rotation matricies can represent a coordiante frame or rotate other frames or vectors, through multiplication')

    # rotation matricies can only rotate, not translate
    st.write('Transformation matricies are 4x4 and can handle translation too')

    # write the formula for the transformation matrix in latex to the streamlit app
    st.latex(r'''
    \begin{align}
    T &= \begin{bmatrix}
    R & p \\
    0 & 1
    \end{bmatrix} \\
    \end{align}
    ''')
            
    st.write('The translation vector, p, is the location of the origin of the new coordinate frame in the old coordinate frame')

    # the formula for the location, p is the translation vector
    st.latex(r'''
    \begin{align}
    p &= \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix} \\
    \end{align}
    ''')
        
# write how this web app was made by Brian Lesko in python to the streamlit app using the libraries streamlit, numpy, and matplotlib
st.write('  ')
st.write('  ')
st.write('  ') 
st.write('Hey Its Brian, thanks for checking out my web app on transformation matricies. I hope you found it useful. Im open for collaboration, reach out to me on LinkedIn or follow me on Github.')
st.write('  ') 
st.write('https://twitter.com/BrianJosephLeko | https://www.linkedin.com/in/brianlesko/ | https://github.com/BrianLesko')

# TO RUN: streamlit run animation.py

