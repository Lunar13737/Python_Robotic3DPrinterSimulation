import numpy as np
import copy
import time
import math
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc
import matplotlib.animation as animation
import timeit

t_0 = timeit.default_timer()

plt.rcParams['animation.embed_limit'] = 2**128
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
})

# Jupyter Notebook
# Uncomment the line below
#from IPython.display import HTML

Lambda = np.array([0.2, -0.2, 10, -1.2])

thetad = np.array([0.2,-0.2,10]) # angular velocity [rad/s], (theta dot)
delv = np.array([0,-1.2,0])   # relative extrusion velocity [m/s]

##-----------------------------------------------------------------------##
## Full Model with Electric, Drag, adn Gravitational  Forces
##-----------------------------------------------------------------------##

## Time Variables
T = float(3)  # total simulation time [s], exclude t=0
dt = float(0.001) # time step size [s]
Nt = int(T/dt) # number of time steps 

## Bed + Arm Variables
Lbed = 0.8   # charged grid side length [m]
r0 = np.array([0,0.5,0])     # fixed arm end positions [m]
theta0 = np.array([np.pi/2,0,0]) # initial rod angles [rad]
Lrod = np.array([0.3,0.2,0.08])   # rod lengths [m]

## Droplet Variables
Nd = Nt  # number of droplets
R = float(0.0001)    # droplet radius [m]
Vi = (4 * np.pi * (R ** 3)) / 3   # droplet volume [m3] 
v2 = 0.25   # phase 2 volume fraction
rho1 = 2000 # phase 1 density [kg/m3] 
rho2 = 7000 # phase 2 density [kg/m3]
q1 = 0   # phase 1 charge capacity [C/m3]
q2 = 0   # phase 2 charge capacity [C/m3]

rhos = rho1 * (1 - v2) + rho2 * v2        # effective density [kg/m3] 
qs = (1 - v2) * q1 +v2 * q2          # effective charge capacity[C/m3]
qi = Vi * qs          # droplet charge [C]
mi = Vi * rhos          # droplet mass [kg]
g = -9.81           # gravitational acceleration [m/s2] 
Fg = np.array([0,g * mi,0])          # gravitational force [N]
Nt = round(1.2*Nt) # add time for particle settling

## E&M Variables
eps = float(8.854E-12) # electric permittivity [F/m] 
qp = float(-8E-5)  # grid pixel charge [C]

## Drag Variables
vf = np.array([0.5,0,0.5])   # surrounding medium velocity [m/s] 
rhoa = 1.225 # surrounding medium density [kg/m3]
muf = float(1.8E10-5)  # surrounding medium viscosity [Pa/s]
Ai = float(np.pi * (R ** 2))   # drag reference area [m2]

## Grid Variables
xgrid = np.linspace(-Lbed/2, Lbed/2, 10)            # grid spacing along x
zgrid = xgrid                                       # grid spacing along z
[Xgrid, Zgrid] = np.meshgrid(xgrid, zgrid)          # create grid of point charges 
Ygrid = np.zeros([10, 10])                          # initally, arms in xz-plane
rp = np.hstack([Xgrid.reshape([np.size(Xgrid),1]),\
                Ygrid.reshape([np.size(Ygrid),1]),\
                Zgrid.reshape([np.size(Zgrid),1])]) # point charge positions [m]

## Initialize Values
theta = theta0
xd = []
yd = [] 
zd = []
r_drop = np.empty([Nd, 3]) 
v_drop = np.empty([Nd, 3]) 
r_disp = np.empty([1,  3])
v_disp = np.empty([1,  3])
F_tot  = np.empty([1,  3]) 
F_elec = np.empty([1,  3]) 
F_drag = np.empty([1,  3])

# For Plotting
plot_ind = 0 #initialize plot index 
drop_plot = list()
Active = np.array([False for i in range(Nd)]) # Logical array to indicate what droplets are still in flight
floor_plot = list()
r_drop = np.ones([Nd,3])
v_drop = np.empty([Nd,3])
r_arm = list()
gInd = []

##-----------------------------------------------------------------------##
## Robotic Arm Dynamics ##
##-----------------------------------------------------------------------##
for i in range(Nt): 
    
    if i < Nd: # only activate droplets as they come out of the dispenser
        Active[i] = True
    indAct = np.where(Active) # indices for active droplets
    indAct = indAct[0][:]
    
    #-------------------------#
    ### Robotc Arm Dynamics ###
    #-------------------------#
    if i < Nd:
        
        ## Dispenser Position
        theta = theta + thetad*dt
        xd = (Lrod[0] * math.cos(theta[0]) 
              + Lrod[1] * math.cos(theta[1]) 
              + Lrod[2] * math.sin(theta[2]))
        yd = (Lrod[0] * math.sin(theta[0]) 
              + Lrod[1] * math.sin(theta[1]))
        zd = Lrod[2] * math.cos(theta[2])
        r_disp = r0 + np.array([xd, yd, zd])
        r_drop[i,:] = r_disp.reshape(3,)

        ## Robot Arm Position
        x_a1 = r0[0] + Lrod[0]*np.cos(theta[0])
        y_a1 = r0[1] + Lrod[0]*np.sin(theta[0])
        x_a2 = x_a1 + Lrod[1]*np.cos(theta[1])
        y_a2 = y_a1 + Lrod[1]*np.sin(theta[1])
        xa = np.array([0, x_a1, x_a2, r_disp[0]]) # x-coord of robot arm
        ya = np.array([0, y_a1, y_a2, r_disp[1]]) # y-coord of robot arm
        za = np.array([0, 0,    0,    r_disp[2]]) # z-coord of robot arm

        ## Dispenser Velocity
        xd_d = (- Lrod[0] * math.sin(theta[0]) * thetad[0] 
                - Lrod[1] * math.sin(theta[1]) * thetad[1]
                + Lrod[2] * math.cos(theta[2]) * thetad[2])
        yd_d = (Lrod[0] * math.cos(theta[0]) * thetad[0] 
                + Lrod[1] * math.cos(theta[1]) * thetad[1])
        zd_d = - Lrod[2] * math.sin(theta[2]) * thetad[2]
        v_disp = np.array([xd_d,yd_d,zd_d]) 
        v_drop[i,:] = v_disp + delv
        
        #----------------------#
        ### Droplet Dynamics ###
        #----------------------#

        ## E&M Force
        F_elec = np.zeros([r_drop[Active,:].shape[0],3]) # reset elec force
        for k in range(F_elec.shape[0]):
            for m in range(rp.shape[0]):
                F_elec[k,:] += (rp[m]-r_drop[Active,:][k,:])*(qp*qi
                                /((4*np.pi*eps)
                                  *((np.linalg.norm(rp[m]-r_drop[Active,:][k,:]))**3)))

        ## Drag Force
        Re = np.array([((2*R*rhoa/muf)*np.linalg.norm(vf - v_drop[Active,:][l,:])) 
                       for l in range(v_drop[Active,:].shape[0])])
        F_drag = np.zeros([len(Re),3]) # reset drag force
        for k in range(len(Re)):
            if Re[k] < 1:
                CDi = Re[k]/24
            elif Re[k] <= 400:
                CDi = 24/Re[k]**0.646
            elif Re[k] <= 3e5:
                CDi = 0.5
            elif Re[k] <= 2e6:
                CDi = 0.000366*Re[k]**0.4275
            else:
                CDi = 0.18
            F_drag[k,:] = (vf-v_drop[Active,:][k,:])*((rhoa*CDi*Ai/2)
                                                      *np.linalg.norm(vf-v_drop[Active,:][k,:]))

        ## Total Force
        F_tot = Fg + F_drag + F_elec

        ## Forward Euler Time Stepping
        # update active matrix
        r_drop[Active,:] = r_drop[Active,:] + v_drop[Active,:] * dt
        v_drop[Active,:] = v_drop[Active,:] + F_tot * (dt / mi)

        # Index droplets that hit ground and deactivate them
        gInd = (r_drop[:,1] <= 0)
        r_drop[gInd,1] = 0
        Active[gInd] = False
        v_drop[gInd,:] = 0

        if i % 50 == 0:
            print(i, 'out of', int(Nt/1.2))
            if i < Nd:
                drop_plot.append(r_drop[Active,:])
            else:
                drop_plot.append(r_drop)
                
            r_arm.append(np.vstack([xa, ya, za]).T)
            if np.size(r_drop[gInd,:]) != 0:
                floor_plot.append(r_drop[gInd,:])
            else:
                floor_plot.append(np.ones([1,3])*1000)

##-----------------------------------------------------------------------##
## Plotting ##
##-----------------------------------------------------------------------##                

def update(n):  # Function to create plot
    dots1.set_xdata(drop_plot[n][:,0])
    dots1.set_ydata(drop_plot[n][:,2])
    dots1.set_3d_properties(drop_plot[n][:,1])
    
    dots2.set_xdata(r_arm[n][:, 0])
    dots2.set_ydata(r_arm[n][:, 2])
    dots2.set_3d_properties(r_arm[n][:, 1])
    
    dots3.set_xdata(r_arm[n][:, 0])
    dots3.set_ydata(r_arm[n][:, 2])
    dots3.set_3d_properties(r_arm[n][:, 1])

    dots4.set_xdata(floor_plot[n][:,0])
    dots4.set_ydata(floor_plot[n][:,2])
    dots4.set_3d_properties(floor_plot[n][:,1])
    
    return dots1, dots2, dots3, dots4

# Set up animation
fig = plt.figure(figsize=(15, 12), dpi=72)
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X [m]', fontsize=20)
ax.set_ylabel('Z [m]', fontsize=20)
ax.set_zlabel('Y [m]', fontsize=20)
ax.set_xlim((-Lbed/2, Lbed/2))
ax.set_ylim((-Lbed/2, Lbed/2))
ax.set_zlim((0, 2))
ax.view_init(elev=15., azim=45)
Title = ax.set_title('Robotic 3D Printer', fontsize=24)
dots1, = ax.plot([], [], [], 'b.', markersize=10)
dots2, = ax.plot([], [], [], 'r-', markersize=5)
dots3, = ax.plot([], [], [], 'r.', markersize=10)
dots4, = ax.plot([], [], [], 'g.', markersize=10)

anim = animation.FuncAnimation(fig, update, frames=len(r_arm), interval=50, blit=False)
writer = animation.writers['ffmpeg']()
anim.save("robotic_3D_printer.mp4", writer=writer, dpi=72)

t_f = timeit.default_timer()
runTime = (t_f - t_0)/60
print("run time = %.2f min" % runTime)
