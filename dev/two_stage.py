import numpy as np
import pyvista as pv
import scipy.integrate as sci
import matplotlib.pyplot as plt


from tqdm import tqdm
from pyvista import examples

import sys
np.set_printoptions(threshold=sys.maxsize)

rplanet = 6371000
mplanet = 5.972e24
G = 6.6742e-11

r = rplanet + 0
period = 16000
# period = np.sqrt((4*np.pi**2 * r**3)/ (self.G * self.mplanet)) 

# FALCON 9 - Somewhat falsified
max_thrust1 = 7427000 # newtons
Isp1 = 283.0 # seconds
tMECO = 162.0 # seconds
tsep1 = 3.0
mass1 = 25600
ve1 = Isp1*9.81 # exit velocity in m/s

max_thrust2 = 794500 # newtons
Isp2 = 348 # seconds
tSECO = tMECO + tsep1 + 397 # seconds
ve2 = Isp2*9.81

def _gravity(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)

    xddot = - G * mplanet * x/ ((r**3)) 
    yddot = - G * mplanet * y/ ((r**3)) 
    zddot = - G * mplanet * z/ ((r**3))

    return np.array([xddot, yddot, zddot])


def _thrust(t):

    theta = 165.0 * np.pi/180 * (1-np.exp(-0.002 * (t)))
    # theta = 0.0

    if t < tMECO:
        thrustF = max_thrust1 #* (1-np.exp(-0.5 * t)) gradient 
        mdot = -thrustF / ve1
    if (t > tMECO) and (t < tMECO + tsep1):
        thrustF = 0.0
        mdot = -mass1 / tsep1
    if (t > tMECO + tsep1):
        thrustF = max_thrust2
        mdot = -thrustF / ve2
    if (t > tSECO):
        thrustF = 0.0
        mdot = 0.0

    thrustx = thrustF * np.cos(theta)
    thrusty = thrustF * np.sin(theta)
    thrustz = 0.0


    return np.array([thrustx, thrusty, thrustz]), mdot
        

def _derivative(t, state_arr, pbar, timing):

    x, y, z, xdot, ydot, zdot, mass = state_arr[:7]

    # if mass < 100:
    #     mdot = 0
    
    gravF = _gravity(x, y, z) * mass
    thrustF, mdot = _thrust(t) 

    Forces = gravF + thrustF

    if mass < 0:
        mdot = 0
        ddot = np.array([0.0, 0.0, 0.0])
    ddot = Forces / mass

    statedot = np.array([xdot, ydot, zdot, ddot[0], ddot[1], ddot[2], mdot])

    last_t, dt = timing
    n = int((t - last_t)/dt)
    pbar.update(n)
    timing[0] = last_t + dt * n

    return statedot


def _solve(state_initial):

    t = np.linspace(0, period, 10000)
    t0 = t[0]
    tf = t[-1]

    with tqdm(total=10000, unit=" iterations") as pbar:
        stateout = sci.solve_ivp(_derivative, [t0, tf], state_initial, t_eval=t, max_step=0.5, args=[pbar, [t0, (tf-t0)/10000]])

    state = stateout.y

    return stateout.y

def _plot_orbit(state):

    plotter = pv.Plotter(window_size=[2100,1400])
    plotter.set_background('black')
    Earth = examples.planets.load_earth(radius=6371000)
    earth_texture = examples.load_globe_texture()
    plotter.add_mesh(Earth, texture=earth_texture, opacity=0.5)

    orbit_points = np.column_stack((state[0,:], state[1,:], state[2,:]))
    # orbit_points = np.column_stack((state[:,0], state[:,1], state[:,2]))
    plotter.add_mesh(orbit_points, color='white', label='Orbit Trajectory', style='wireframe', point_size=1.5)

    plotter.add_legend(size=(0.15, 0.15), face=None, bcolor='black')

    _ = plotter.add_axes(
        color='white',
        line_width=3,
        cone_radius=0.2,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )

    plotter.show()


def _plot_states(state):
        v = np.sqrt(state[3,:]**2 + state[4,:]**2 + state[5,:]**2)
        r = np.sqrt(state[0,:]**2 + state[1,:]**2 + state[2,:]**2) - rplanet
        mass = state[6,:]
        t = np.linspace(0, period, 10000)

        index_a = np.argmax(r)
        index_p = np.argmin(r[1000:]) + 1000

        index_m = np.argmin(mass)
        print(min(mass))

        inclin = np.rad2deg(np.arctan2(state[2,:], np.sqrt(state[0,:]**2 + state[1,:]**2)))

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(25, 14))
        fig.suptitle(f"Launch")
        ax0.plot(t, mass, label='Mass', c='red')
        ax0.set_title("Mass vs Time")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Mass")
        ax0.annotate(f"Mass Min: {mass[index_m]:,.0f} kg", 
            xy=(t[index_m], mass[index_m]),
            xytext=(-120,0),
            textcoords='offset points', 
            arrowprops=dict(arrowstyle='->', 
                            connectionstyle='arc3,rad=0.3',     
                            color='black', 
                            linewidth=1.5)
                    )

        ax1.plot(t, r, label='Position', c='blue')
        ax1.set_title("Position vs Time")
        ax1.annotate(f"Apogee: {r[index_a]:,.0f} m", 
            xy=(t[index_a], r[index_a]),
            xytext=(-120,0),
            textcoords='offset points', 
            arrowprops=dict(arrowstyle='->', 
                            connectionstyle='arc3,rad=0.3',     
                            color='black', 
                            linewidth=1.5)
                    )
        ax1.annotate(f"Perigee: {r[index_p]:,.0f} m", 
            xy=(t[index_p], r[index_p]),
            xytext=(-125, -5),
            textcoords='offset points', 
            arrowprops=dict(arrowstyle='->', 
                            connectionstyle='arc3,rad=0.3',     
                            color='black', 
                            linewidth=1.5)
                    )
        plt.show()


        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(25, 14))
        fig.suptitle(f"Launch")
        ax0.plot(t, v, label='Velocity', c='red')
        ax0.set_title("Velocity vs Time")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Velocity")

        plt.show()

x0 = rplanet
y0 = 0.0
z0 = 0.0
xv0 = 0.0
yv0 = 0.0
zv0 = 0.0
mass0 = 565961.77
prop_mass0 = 395700

state_initial = ([x0, y0, z0, xv0, yv0, zv0, mass0])

stateout = _solve(state_initial)
_plot_orbit(stateout)
_plot_states(stateout)











# def _derivative(t, state_arr):

#     x, y, z, xdot, ydot, zdot, mass = state_arr[:7]

#     # if mass < 100:
#     #     mdot = 0
    
#     gravF = _gravity(x, y, z) * mass
#     thrustF, mdot = _thrust(t) 

#     Forces = gravF + thrustF

#     if mass < 0:
#         mdot = 0
#         ddot = np.array([0.0, 0.0, 0.0])
#     ddot = Forces / mass

#     statedot = np.array([xdot, ydot, zdot, ddot[0], ddot[1], ddot[2], mdot])

#     return statedot

# from utils import rk45


# tarr = np.linspace(0, 1000, 50000)
# ti = tarr[0]

# dt = 1000 / 20000

# t0 = tarr[0]
# tf = tarr[-1]

# state0 = ([x0, y0, z0, xv0, yv0, zv0, mass0])
# with tqdm(total=10000, unit=" iterations") as pbar:
#     stateout = rk45.rk45(_derivative, tarr, state0, args=[pbar, [t0, (tf-t0)/10000]])

# print(stateout[5,:])
# _plot_orbit(stateout)
# _plot_orbit(stateout)

# k1 = _derivative(ti, state0)
# k2 = _derivative(ti + dt/4, state0 + k1*dt/4)
# k3 = _derivative(ti + dt/4, state0 + k2*dt/4)
# k4 = _derivative(ti + dt/2, state0 + k3*dt/2)
# final = state0 + dt/12*(k1 + 2*k2 + 2*k3 + k4)
# print(final)
# k1 = _derivative(ti, final)
# k2 = _derivative(ti + dt/4, final + k1*dt/4)
# k3 = _derivative(ti + dt/4, final + k2*dt/4)
# k4 = _derivative(ti + dt/2, final + k3*dt/2)
# test = final + dt/12*(k1 + 2*k2 + 2*k3 + k4)
# print(test)
# print(stateout[:, 1])
