# Pendulum Bowl generator thing by Danny Walker, 2018
# http://danny.makesthings.work || https://walkerdanny.github.io
# CC BY-NC-SA 4.0
# Soon to be a major motion picture

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # The 3D plot is useful to preview the shape without rendering the OpenSCAD model
import numpy as np


## INTITIAL CONDITIONS!

num_points = 1000 # Number of timesteps to simulate
fn = 20 # The "fineness" of the curved shapes in OpenSCAD. Higher is better, but greatly increases render time. 20 is good for me.

# Minimum and maximum radii of the spheres which draw the path. Tweak these to whatever your printer can comfortably handle and what looks good
minRad = 0.5
maxRad = 2.2

L = 80 # Length of the pendulum
g = -50 # Gravity. I know, I'm sorry.
friction = 0.999 # Viscous friction. Velocities multiply by this each timestep. Set to 1 for frictionless.

theta_in = 90 # Initial starting angle. Theta is the angle that traces "up and down the walls of the bowl" or whatever
theta_v_in = 0 # Initial starting angular velocity in theta. Zero is equivalent to dropping it, anything else and you're giving it a shove
theta_a_in = 0 # Initial acceleration, not much point in using it as it'll be overwritten on the first timestep

phi_in = 0 # Initial angle "around the edge of the bowl". Kind of pointless to set it as you'll be getting a 3D model, but whatever
phi_v_in = 0.29 # Speed at which the pendulum moves around the edge of the bowl. Warning: if you set this too large you end up with a ring shape!
phi_a_in = 0 # Again, kind of useless since phi_a is fixed at zero

# DON'T CHANGE ANYTHING BELOW HERE UNLESS YOU KNOW WHAT YOU'RE DOING! Or do! I don't really mind! It might break though.

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1.2*L, 1.2*L])
ax.set_xlabel('X')
ax.set_ylim3d([-1.2*L, 1.2*L])
ax.set_ylabel('Y')
ax.set_zlim3d([-1.2*L, 1.2*L])
ax.set_zlabel('Z')
ax.set_title('Spherical Pendulum')

# Calcualate the Cartesian coordinates of the initial conditions
xPos_in = L*np.sin(np.radians(theta_in))*np.cos(np.radians(phi_in))
yPos_in = L*np.sin(np.radians(theta_in))*np.sin(np.radians(phi_in))
zPos_in = -L*np.cos(np.radians(theta_in))

# Fill up the data array for the inial conditions. This then gets iterated at each timestep.
data = [theta_in, theta_v_in, theta_a_in, phi_in, phi_v_in, phi_a_in, xPos_in, yPos_in, zPos_in]

# Set up a new array to store the traced points (with an extra column for velocity)
thePointz = np.zeros((num_points,4))


def velMag(theta_v_input, phi_v_input): # Function to calculate the magnitude of the angular velocities. Kina obvious.
    mag = np.sqrt(theta_v_input*theta_v_input + phi_v_input*phi_v_input)
    return mag

def linear_scale(variable, minIn, maxIn, minOut, maxOut): # Surely I don't need to implement this myself? Can't numpy already do this?
    # It's only y = mx + c scaling!
    m = (maxOut-minOut)/(maxIn-minIn)
    c = minOut - m * minIn
    return m*variable + c

def update_bob(): # Iteration function to update the data array for each timestep
    # A bob is the thing on the end of a pendulum, I didn't call my data array Bob or anything.
    global data
    global friction

    theta = data[0]
    theta_v = data[1]
    theta_a = data[2]

    phi = data[3]
    phi_v = data[4]
    phi_a = data[5]

    # Equations of motion of the system:
    theta_a = (np.sin(np.radians(theta))*np.cos(np.radians(theta))*phi_v*phi_v) - ((g/L)*np.sin(np.radians(theta)))

    # This line is commented out because of... reasons.

    #phi_a = -2*phi_v*theta_v*(np.cos(np.radians(theta))/np.sin(np.radians(theta)))
    # If you actually implement the acceleration in phi, bear in mind it tends to infinity as theta approaches zero, and you should deal with this.
    # My solution is to ignore it and pretend phi_v is constant

    # This is a really basic physics engine
    phi_a = phi_a_in
    theta_v += theta_a
    phi_v += phi_a

    # Simulate some viscous friction!
    theta_v *= friction
    phi_v *= friction

    theta += theta_v
    phi += phi_v

    # Calculate the Cartesian coordinates again
    xPos = L*np.sin(np.radians(theta))*np.cos(np.radians(phi))
    yPos = L*np.sin(np.radians(theta))*np.sin(np.radians(phi))
    zPos = -L*np.cos(np.radians(theta))

    # ...and fill up the data array again before returning it
    data = [theta, theta_v, theta_a, phi, phi_v, phi_a, xPos, yPos, zPos]

    return data

# Do the iteration for the amount of timesteps and fill the array up!
for i in xrange(num_points):
    data = update_bob()
    thePointz[i] = [data[6], data[7], -data[8], velMag(data[1], data[4])]

# Plot the data to preview the shape
bob = ax.plot(thePointz[:,0], thePointz[:,1], thePointz[:,2], '-', linewidth=0.5)

# The maximum and minimum velocities during the simulation. Useful for scaling later.
maxVel = np.amax(thePointz[:,3])
minVel = np.amin(thePointz[:,3])

# Okay, from here on is actually pretty janky and I can only apologise.
# Basically, what's going to happen is it's going to draw a sphere with a radius proportional to the pendulum's
# velocty at each coordinate in the array. It's then going to do the same for the next data point, and then
# perform a hull() command between the two. This generates a good-looking, but really slow OpenSCAD model.
# Also I'm doing it completely by concatenating strings, which is Brexit, but it works.
# Sorry it's gross to read.

openscad_string = ''
for i in xrange(num_points-1):
    thisPoint = thePointz[i]
    nextPoint = thePointz[i+1]
    # Linear interpolation to calculate the radii as a function of the velocity
    r1 = linear_scale(thisPoint[3], minVel, maxVel, minRad, maxRad)
    r2 = linear_scale(nextPoint[3], minVel, maxVel, minRad, maxRad)

    # Eugh...
    openscad_string += 'hull($fn=%(fn)f){\n' % {'fn':fn}
    openscad_string += '\ttranslate([%(xZero)f, %(yZero)f, %(zZero)f]){\n' % {"xZero":thisPoint[0], "yZero": thisPoint[1], "zZero": thisPoint[2]}
    openscad_string += '\t\tsphere(%(rad)f, $fn=%(fn)f);\n' % {'rad':r1, 'fn':fn}
    openscad_string += '\t}\n'
    openscad_string += '\ttranslate([%(xZero)f, %(yZero)f, %(zZero)f]){}\n' % {"xZero":-thisPoint[0], "yZero": -thisPoint[1], "zZero": -thisPoint[2]}
    openscad_string += '\ttranslate([%(xOne)f, %(yOne)f, %(zOne)f]){\n' % {"xOne":nextPoint[0], "yOne": nextPoint[1], "zOne": nextPoint[2]}
    openscad_string += '\t\tsphere(%(rad)f, $fn=%(fn)f);\n' % {'rad':r2, 'fn':fn}
    openscad_string += '\t}\n'
    openscad_string += '\ttranslate([%(xOne)f, %(yOne)f, %(zOne)f]){}\n' % {"xOne":-nextPoint[0], "yOne": -nextPoint[1], "zOne": -nextPoint[2]}
    openscad_string += '};\n'

# This line adds on a circular "base" to the bottom of the "bowl". You might not want this, but it helps it print and stand up and stuff.
openscad_string += 'translate([%(xThree)f, %(yThree)f, %(zThree)f]){\n\tcylinder(h=5, r1 = %(r)f, r2 = %(r)f, center=true, $fn=%(fn)f);\n};' % {'xThree': 0, 'yThree': 0,'zThree':-L, 'r': L/3,'fn': 100}

# Generate the .scad file!
cadFile = open('output.scad','w')
cadFile.write(openscad_string)
cadFile.close()

# Show the plot!
plt.show()
