import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from pprint import pprint
from typing import Optional
from pyvista import examples

from src.satellite import launch_pos as lp
from src.satellite import sat_orbit as sat
from src.utils import data 
from src.utils import states

class Plotter():

    def __init__(
        self,
        data,
        args: Optional[dict] = None,
        labels: Optional[tuple] = None
    ):

        _args = {
            'ax_size': (1, 2),
            'ax_figs': ['velocity', 'position'],
            'background': 'white',
            'fig_size': (24, 15),
            'legend': True,
            'multi_array': False,
            'radius': None,
            'show': True,
            'show_planets': False,
            'title': 'Space',
            'xlabel': ['Time', 'Time'],
            'ylabel': ['Velocity', 'Position'],
            'launch': None,
            'target': None,
            'target_str': 'Set Target Str'
        }

        if args is not None:
            for key in args.keys():
                _args[key] = args[key]

        self.data = data
        
        if labels is not None:
            self.labels = labels

        self.args = _args

    def print__args(self):
        pprint(self.args)

    def solar_system(self):
        """ Plots the position array passed with associated labels and 
            appropriate colors.

            Args:
                pos_arrays: tuple or array
                            an array or tuple of arrays each containing all information
                            needed to plot the orbit
                labels: tuple or array
                        list of labels that line up with each position array for
                        adding an accurate legend

        """

        if self.args['multi_array'] == True:
            pos_arrays = self.data
            labels = self.labels
            assert len(pos_arrays) == len(labels), "must have the same number of labels as you have planets"
        else:
            pos_arrays = [self.data]
            labels = [self.labels]

        plotter = pv.Plotter()

        if self.args['show_planets']:
            show_planet = True
        else:
            show_planet = False
        
        if len(pos_arrays) > 1:

            for index, arr in enumerate(pos_arrays):
                
                label = labels[index].lower()
                color = data.color_map.get(label)

                r = self.args[ 'radius' ][index]
                print(r)
                print(arr)

                polydata = pv.Sphere(
                    radius = self.args[ 'radius' ][index], 
                    center=np.array(arr)
                )
                plotter.add_mesh(polydata, color=color, point_size=3, label=labels[index].title(), render_points_as_spheres=True)
                if show_planet:
                    planet = pv.PolyData(arr[-1])
                    plotter.add_mesh(planet, color=color, point_size=10, render_points_as_spheres=True)
        else:

            polydata = pv.PolyData(pos_arrays)
            plotter.add_mesh(polydata, color='blue', point_size=2, label=labels[0].title())

        sun_center = np.array([0, 0, 0])

        # Sun radius in meters but the plots are so small Earth isn't visible at all
        # sun_radius = 696340000

        sun_radius = 6963400
        sun_sphere = pv.Sphere(radius=sun_radius, center=sun_center)
        plotter.add_mesh(sun_sphere, color='orange', label='Sun')

        plotter.camera_position = 'iso'
        plotter.camera.zoom(0.8)

        plotter.set_background(self.args['background'])

        if self.args[ 'legend' ]:
            plotter.add_legend(bcolor=(255, 255, 255), border=False, size=[0.15, 0.15], face=None)

        if self.args[ 'show' ]:
            plotter.show()

    def satellite(self):
        plotter = pv.Plotter(window_size=[2100,1400])
        plotter.set_background('black')
        Earth = examples.planets.load_earth(radius=6371000)
        earth_texture = examples.load_globe_texture()
        plotter.add_mesh(Earth, texture=earth_texture)

        orbit_points = np.column_stack((self.data[0,:], self.data[1,:], self.data[2,:]))
        plotter.add_mesh(orbit_points, color='white', label='Orbit Trajectory', style='wireframe', point_size=1.5)

        if (self.args['launch'] is not None):
            launch_point = pv.PolyData([self.args['launch'][0], self.args['launch'][1], self.args['launch'][2]])
            plotter.add_mesh(launch_point, color='yellow', point_size=15, render_points_as_spheres=True, label='Launch')     

        if (self.args['target'] is not None):
            target_point = pv.PolyData([self.args['target'][0], self.args['target'][1], self.args['target'][2]])
            plotter.add_mesh(target_point, color='#89CFF0', point_size=15, render_points_as_spheres=True, label=self.args['target_str'].title())     
        
        if (self.args['legend']):
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
        if (self.args['show']):
            plotter.show()

    def states(self):

        plt.close()

        assert (self.args['ax_size'][0]*self.args['ax_size'][1]) == len(self.args['ax_figs'])
        
        fig, ax = plt.subplots(self.args['ax_size'][0], self.args['ax_size'][1], figsize=self.args['fig_size'])
                
        state_arr = states.get_states(self.data, self.args['ax_figs'])  

        if np.ndim(ax) == 1:
            for row in range(len(ax)):
                ax[row].plot(range(len(state_arr[0])), state_arr[row])
                ax[row].set_xlabel(self.args['xlabel'][row])
                ax[row].set_ylabel(self.args['ylabel'][row])
        else:
            for row in range(len(ax)):
                for column in range(len(ax[0])):
                    ax[row][column].plot(range(len(state_arr[row+column])), state_arr[row+column])
                    ax[row][column].set_xlabel()
                    ax[row][column].set_title(self.args['ax_figs'][row+column].title())

        
        fig.suptitle(self.args['title'])

        plt.show()
    """
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
    """

if __name__ == "__main__":
    # SET DESIRED ORBIT HEIGHT > 99000m
    orbit_height = 100000.0#m

    # Comparing with Rocket Lab's Electron 47 launch which had a starting velocity of 28570 km/h
    # for an elliptical orbit. This gave initial velocity of ~28000 km/h. Starting positions and 
    # orbital path differ, however the similarity is encouraging.
    # orbit_height = 210000.6

    r = float(c.rplanet + orbit_height)

    # Set desired target position
    country = 'united states'
    city = 'salt lake city'

    # This gives us our starting positions and velocities
    get_pos = lp.LaunchPosition(country, city, r, c.G, c.mplanet)
    state_initial = get_pos._get_launch_pos()
    target = get_pos.target_pos

    # # Pass starting positions and velocities into SatelliteOrbit class
    orbit = sat.SatelliteOrbit(state_initial, orbit_height, 0.6, city, anomoly=2)
    
    stateout = orbit._solve()
    # orbit._plot_states()

    plot = Plotter(data=stateout)
    plot.satellite()
