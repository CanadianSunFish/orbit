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

import sys
sys.path.append('../src')

import os
print(os.getcwd())

# Constants file
from utils import constants as c

# For loading bar
from tqdm import tqdm

class SatelliteOrbit():

    def __init__(
        self,
        statein: Union[tuple, list, np.array],
        height: Union[int, float],
        mass: Union[int, float],
        target: str,  
        G: Optional[float] = c.G,
        mplanet: Optional[float] = c.mplanet,
        rplanet: Optional[float] = c.rplanet,
        anomoly: Optional[Union[int, float]] = 1
    ):
        """ Construct `Satellite Orbit`

            Args:
                pos:
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

        assert isinstance(statein[0], float), 'x pos must be float (if dealing with ints type cast or add .0)'
        self.x0 = statein[0]
        assert isinstance(statein[1], float), 'y pos must be float (if dealing with ints type cast or add .0)'
        self.y0 = statein[1]
        assert isinstance(statein[2], float), 'z pos must be float (if dealing with ints type cast or add .0)'
        self.z0 = statein[2]

        assert isinstance(statein[3], float), 'xdot must be float (if dealing with ints type cast or add .0)'
        self.xv0 = statein[3]
        assert isinstance(statein[4], float), 'ydot must be float (if dealing with ints type cast or add .0)'
        self.yv0 = statein[4]
        assert isinstance(statein[5], float), 'zdot must be float (if dealing with ints type cast or add .0)'
        self.zv0 = statein[5]

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

        self.state_initial = np.array([self.x0, self.y0, self.z0, self.xv0, self.yv0, self.zv0])

        print(self.G*self.mplanet)

    def __str__(self):
        v = np.sqrt(self.xv0**2 + self.yv0**2 + self.zv0**2)
        return (f"""
Target: {self.target}

Orbital Attributes: 
height: {self.height:,.0f} m ({self.height/1000:,.0f} km)
period: {self.period0/60:.1f} min ({self.period0/60/60:.2f} hours)

Starting Positions:
x: {self.x0:,.0f} m
y: {self.y0:,.0f} m
z: {self.z0:,.0f} m

Starting Velocities:
xdot: {self.xv0:,.2f} m/s
ydot: {self.yv0:,.2f} m/s
zdot: {self.zv0:,.2f} m/s
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

    def _plot_orbit(self, launch, target):
        """ Plots the orbit using the instance variable stateout. Also
            takes arguments for launch and target positions for visual
            effect. 

            Args:
                launch: float array
                    x y z coordinate array representing 'launch' position
                target: float array
                    x y z coordinate array representing 'target' position
        
        """

        plotter = pv.Plotter(window_size=[2100,1400])
        plotter.set_background('black')
        Earth = examples.planets.load_earth(radius=6371000)
        earth_texture = examples.load_globe_texture()
        plotter.add_mesh(Earth, texture=earth_texture)

        orbit_points = np.column_stack((self.state[0,:], self.state[1,:], self.state[2,:]))
        plotter.add_mesh(orbit_points, color='white', label='Orbit Trajectory', style='wireframe', point_size=1.5)

        target_point = pv.PolyData([target[0], target[1], target[2]])
        launch_point = pv.PolyData([launch[0], launch[1], launch[2]])

        plotter.add_mesh(target_point, color='#89CFF0', point_size=15, render_points_as_spheres=True, label=self.target)     
        plotter.add_mesh(launch_point, color='yellow', point_size=15, render_points_as_spheres=True, label='Launch')     
        
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
        #Not currently in use
        #plotter.screenshot(f'./src/figs/{self.target.lower()}_orbit.png')


    def _plot_states(self):
        v = np.sqrt(self.state[3,:]**2 + self.state[4,:]**2 + self.state[5,:]**2)
        r = np.sqrt(self.state[0,:]**2 + self.state[1,:]**2 + self.state[2,:]**2) - self.rplanet
        t = np.linspace(0, self.period, 10000)

        index_a = np.argmax(r)
        index_p = np.argmin(r)

        inclin = np.rad2deg(np.arctan2(self.state[2,:], np.sqrt(self.state[0,:]**2 + self.state[1,:]**2)))

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(25, 14))
        fig.suptitle(f"Data for {self.target.title()} Orbit")
        ax0.plot(t, v, label='Velocity', c='red')
        ax0.set_title("Velocity vs Time")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Velocity")

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
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position")

        ax2.plot(t, inclin)
        ax2.set_title("Inclination vs Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Inclination (degrees)")

        theta = np.rad2deg(np.arctan2(self.state[0,:], self.state[1,:]))
        ax3.plot(t, theta)
        ax3.set_title("Theta vs Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("${\Theta}$")

        # plt.savefig('./src/figs/state_plot.png')
        plt.show()

class LaunchPosition():

    def __init__(
        self,
        country: str,
        city: Optional[str],
        r: int,
        G: float,
        mplanet: float,
        output: Optional[bool] = False
    ):  
        """ Construct `LaunchPosition`

            Args:
                country: str
                    target country
                city: Optional str
                    city in country for target orbit. If no 
                    city is passed, largest city in target 
                    country will be used
                r: float
                    radius of orbit from center of planet (in meters). 
                    must be outside the atmosphere, above 6470000m
                G: float
                    gravitational constant of the planet in N*m**2/kg**2
                mplanet: float
                    mass of the planet in kg
                output: Optional bool
                    specifies if printing of city and country 
                    information should occur

        """ 

        assert isinstance(country, str), 'country must be a str'
        self.country = country
        if city is not None:
            assert isinstance(city, str), 'city must be a str'
        self.city = city

        assert isinstance(r, float), 'r must be a float (maybe add a .0)'
        assert r > 6371000 + 99000, f'orbit radius r must be be above the atmosphere {6371000 + 99000}m, currently {r}m'
        self.r = r

        assert isinstance(mplanet, float), 'mplanet must be a float (maybe add a .0)'
        self.mplanet = mplanet
        assert isinstance(G, float), 'G must be a float (maybe add a .0)'
        self.G = G

        assert isinstance(output, bool), 'output must be a bool'
        self.output = output

        self.x, self.y, self.z = self._get_target_orbit()

        self.target_pos = np.array([self.x, self.y, self.z])

    def _get_target_orbit(self):
        """
            Gets the coordinates above the target city.
        """
        df = pd.read_csv('src/utils/data/worldcities.csv')

        country = self.country.title()
        city = self.city.title()

        mask_country = np.column_stack([df['country'].str.contains(country, na=False) for col in df])
        mask_city = np.column_stack([df['city_ascii'].str.contains(city, na=False) for col in df])

        lat = 0
        lng = 0

        try:
            city = df.loc[mask_city.any(axis=1)].iloc[0]
            lat = np.deg2rad(city.lat)
            lng = np.deg2rad(city.lng)

            if self.output == True:
                print(f'Located {city.city}, {city.country}')
                print(f'Lat: {city.lat}, Lng:{city.lng}')
        except:
            try:
                new_city = df.loc[mask_country.any(axis=1)].iloc[0]
                lat = np.deg2rad(new_city.lat)
                lng = np.deg2rad(new_city.lng)

                if self.output == True:
                    print(f'Could not locate {city}, instead using: {new_city.city}, {country}')
                    print(f'Lat: {lat}, Lng:{lng}')
            except:
                print(f'No city in {country} could be found. Try again.')
                quit

        x = self.r * np.cos(lng)*np.cos(lat)
        y = self.r * np.sin(lng)*np.cos(lat)
        z = self.r * np.sin(lat)

        return -x, -y, z

    def _get_launch_pos(self):
        """
            Gets the 'launch' position (initial conditions for the orbit
            to be above the target city)
        """

        phi = np.arctan2(self.y,self.x)

        xtemp = self.r * np.cos(phi)
        ytemp = self.r * np.sin(phi)

        x0 = ytemp
        y0 = -xtemp
        
        z0 = 0.0

        theta = np.linspace(0, 0.1 * np.pi, 1000)
        theta = theta[:2]

        interp_x = np.array([np.cos(the)*x0 + np.sin(the)*self.x for the in theta])
        interp_y = np.array([np.cos(the)*y0 + np.sin(the)*self.y for the in theta])
        interp_z = np.array([np.cos(the)*z0 + np.sin(the)*self.z for the in theta])

        x = interp_x[1] - interp_x[0]
        y = interp_y[1] - interp_y[0]
        z = interp_z[1] - interp_z[0]

        v = np.sqrt(self.mplanet*self.G/self.r)

        r0 = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r0)
        
        xv0 = v * np.cos(phi) * np.sin(theta) 
        yv0 = v * np.sin(phi) * np.sin(theta) 
        zv0 = v * np.cos(theta) 

        return np.array([x0, y0, z0, xv0, yv0, zv0])

if __name__ == "__main__":

    import time

    start = time.time()

    # SET DESIRED ORBIT HEIGHT > 99000m
    orbit_height = 3500000.0#m

    # Comparing with Rocket Lab's Electron 47 launch which had a starting velocity of 28570 km/h
    # for an elliptical orbit. This gave initial velocity of ~28000 km/h. Starting positions and 
    # orbital path differ, however the similarity is encouraging.
    # orbit_height = 210000.6

    r = float(c.rplanet + orbit_height)

    # Set desired target position
    country = 'norway'
    city = 'oslo'

    # This gives us our starting positions and velocities
    get_pos = LaunchPosition(country, city, r, c.G, c.mplanet)
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
    # orbit._plot_orbit(state_initial, target)
    orbit._plot_states()
    
