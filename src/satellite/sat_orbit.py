""" 
    Utilizes central force equations to plot satelite orbit over specified city.
"""
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.integrate as sci
import matplotlib.pyplot as plt

from pyvista import examples
from typing import Optional, Union

# Local imports
from src.utils import states as st
from src.utils import data
from src.utils import plotting_utils as oplt
from src.satellite import launch_pos as pos

# For loading bar
from tqdm import tqdm

class SatelliteOrbit():

    def __init__(
        self,
        statein: Union[tuple, list, np.array],
        height: Union[int, float],
        mass: Union[int, float],
        target: str,  
        G: Optional[float] = data.G,
        mplanet: Optional[float] = data.mplanet,
        rplanet: Optional[float] = data.rplanet,
        anomoly: Optional[Union[int, float]] = 1
    ):
        """ Construct `Satellite Orbit`

            Args:
                statein:
                    An array of all initial values for orbit
                    which includes x0, y0, z0, xv0, yv0, zv0
                    in m for positions and m/s for velocities.
                height: float
                    Desired orbital height in meters above
                    the surface of the planet. 
                mass: float
                    Satellite mass in kg.
                G: float
                    Gravitational constant of the planet in 
                    N*m^2/kg^2
                mplanet: float
                    Mass of the planet in kg.
                rplanet: float
                    Average radius of the planet in m.
                anomoly: float or int
                    Desired portion of the orbit to calculate
                    with 0.1 being 1/10 of an orbit and 5
                    being 5 full orbits. 
        """

        assert all(isinstance(val, float) for val in statein), 'initial conditions must be floats'
        self.state_initial = np.array(statein)

        assert isinstance(height, (int, float)), 'height must be numerical value'
        assert height > 0, 'height must be greater than 0 (must orbit outside the planet)'
        self.height = height

        assert isinstance(G, float), 'G must be a float'
        assert G > 0, 'G (gravitational constant) must be positive'
        self.G = G

        assert isinstance(mass, (int, float)), 'mass must be a numerical value'
        assert mass > 0, 'mass must be positive'
        self.mass = mass

        assert isinstance(target, str), 'target must be a str'
        self.target = target.title()

        assert isinstance(mass, (int, float)), 'mplanet must be a float'
        assert mplanet  > 0, 'mplanet must be positive'
        self.mplanet = mplanet

        assert isinstance(mass, (int, float)), 'rplanet must be a float'
        assert rplanet > 0, 'rplanet must be positive'
        self.rplanet = rplanet

        assert isinstance(anomoly, (int, float)), 'anomoly must be numerical value'
        assert anomoly > 0, 'anomoly must be positive'

        self.r = rplanet + height

        self.period0 = np.sqrt((4*np.pi**2 * self.r**3)/ (self.G * self.mplanet)) 
        self.period = self.period0 * anomoly


    def __str__(self):

        v = st.get_states(self.state_initial, 'velocity')[0]

        return (f"""
Target: {self.target}

Orbital Attributes: 
height: {self.height:,.0f} m ({self.height/1000:,.0f} km)
period: {self.period0/60:.1f} min ({self.period0/60/60:.2f} hours)

Starting Positions:
x: {self.state_initial[0]:,.0f} m
y: {self.state_initial[1]:,.0f} m
z: {self.state_initial[2]:,.0f} m

Starting Velocities:
xdot: {self.state_initial[3]:,.2f} m/s
ydot: {self.state_initial[4]:,.2f} m/s
zdot: {self.state_initial[5]:,.2f} m/s
total: {v:,.2f} m/s ({v*3.6:,.2f} km/h)

Planet Attributes:
mass: {self.mplanet} kg
radius: {self.rplanet:,.0f} m ({self.rplanet/1000:,.0f} km)
G: {self.G} N*m\u00b2/kg\u00b2
        """)

    def _gravity(self, x, y, z):
        """ Calculating the force due to gravity on the object at the 
            given arg position. Uses (- Gplanet * mplanet * rhat / r**3).
            Originally this equation has an r**2 value in the denominator,
            but the rhat value is a scalar and an extra r value needs to 
            be added to the denominator. This requires the args to be 
            representing a position in 3d Cartesian space with an inertial 
            reference frame centered at the coordinates (0, 0, 0).

            Args:
                x: float
                    value representing the x position of the spacecraft
                y: float
                    value representing the y position of the spacecraft
                z: float
                    value representing the z position of the spacecraft

        """

        r = np.sqrt(x**2 + y**2 + z**2)

        xddot = - self.G * self.mplanet * x/ ((r**3)) 
        yddot = - self.G * self.mplanet * y/ ((r**3)) 
        zddot = - self.G * self.mplanet * z/ ((r**3))

        return np.array([xddot, yddot, zddot])

    def _derivative(self, t, state_arr, pbar, timing):
        """ Derivative method for use with Scipy's solve_ivp method. Uses _gravity 
            method to calculate acceleration using Fgrav = m*a -> Fgrav/m = a

            Args:
                t: float array
                    time values that will be used during the calculations
                state_arr: float array
                    representing the initial state of the satellite. this should
                    contain x0, y0, z0, xv0, yv0, and zv0 in that order.
                pbar: tqdm pbar
                    used in creating the progress bar
                timing: optional float array
                    values that will be used in calculating and presenting
                    the progress bar. the array should be the same len as
                    the t array

            Returns:
                statedot: Scipy array
                    representing the t, x, y, z, xv, yv, zv values 
        """

        x, y, z, xdot, ydot, zdot = state_arr[:6]
        
        Forces = self._gravity(x, y, z) * self.mass

        ddot = Forces/self.mass

        statedot = np.array([xdot, ydot, zdot, ddot[0], ddot[1], ddot[2]])

        last_t, dt = timing
        n = int((t - last_t)/dt)
        pbar.update(n)
        timing[0] = last_t + dt * n

        return statedot

    def _solve(self):
        """ Solves the ivp set up by the methods _derivative and _gravity. 

            Returns:
                stateout:
                    an array that holds x, y, z, xv, yv, zv values in that
                    order
        """

        t = np.linspace(0, self.period, 10000)
        t0 = t[0]
        tf = t[-1]

        with tqdm(total=10000, unit=" iterations") as pbar:
            stateout = sci.solve_ivp(self._derivative, [t0, tf], self.state_initial, t_eval=t, max_step=0.5, args=[pbar, [t0, (tf-t0)/10000]])

        self.state = stateout.y

        return stateout.y

if __name__ == "__main__":

    import time

    start = time.time()

    # SET DESIRED ORBIT HEIGHT > 99000m
    orbit_height = 2000000.0#m

    # Comparing with Rocket Lab's Electron 47 launch which had a starting velocity of 28570 km/h
    # for an elliptical orbit. This gave initial velocity of ~28000 km/h. Starting positions and 
    # orbital path differ, however the similarity is encouraging.
    # orbit_height = 210000.6

    r = float(c.rplanet + orbit_height)

    # Set desired target position
    country = 'united states'
    city = 'salt lake city'

    # This gives us our starting positions and velocities
    get_pos = pos.LaunchPosition(country, city, r, data.G, data.mplanet)
    state_initial = get_pos._get_launch_pos()
    target = get_pos.target_pos

    # Pass starting positions and velocities into SatelliteOrbit class
    orbit = SatelliteOrbit(state_initial, orbit_height, 0.6, city, anomoly=2)

    # Print class atributes
    print(orbit)

    # Solve the ode
    stateout = orbit._solve()

    end = time.time()
    run = end-start
    print(f'Run time: {run:.2f}s')

    # Plot orbit and states
    plot = oplt.Plotter(data=stateout, args={'launch':state_initial, 'target':target, 'target_str':city})
    plot.satellite()
    # plot.states()

    
