# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import pyvista as pv

# EC = 9.327732834703664E-02
# A = 2.279312392979201E+08
# IN = np.radians(1.847853958869473E+00)
# OM = np.radians(4.948926499126916E+01)
# W = np.radians(2.866748818306182E+02)

# mars_ec = 9.327732834703664E-02
# mars_om = np.radians(4.948926499126916E+01)
# mars_in = np.radians(1.847853958869473E+00)
# mars_w = np.radians(2.866748818306182E+02)
# mars_a = 2.279312392979201E+08

# earth_ec = 1.671168717375127E-02
# earth_om = np.radians(1.317874386226365E+02)
# earth_in = np.radians(1.467815966965072E-03)
# earth_w = np.radians(3.343857726397638E+02)
# earth_a = 1.497390619236750E+08

# # Calculate the position of the body at various true anomaly values
# def plot_data(EC, OM, A, IN, W):
#     true_anomalies = np.linspace(0, 2*np.pi, 1000)
#     positions = []

#     for ta in true_anomalies:
#         # Calculate the distance from the Sun to the body
#         r = A * (1 - EC**2) / (1 + EC * np.cos(ta))
        
#         # Calculate the position in the perifocal coordinate system
#         x_p = r * np.cos(ta)
#         y_p = r * np.sin(ta)
        
#         # Convert the position to the geocentric equatorial coordinate system
#         x = (np.cos(OM) * np.cos(W) - np.sin(OM) * np.sin(W) * np.cos(IN)) * x_p + \
#             (-np.cos(OM) * np.sin(W) - np.sin(OM) * np.cos(W) * np.cos(IN)) * y_p
#         y = (np.sin(OM) * np.cos(W) + np.cos(OM) * np.sin(W) * np.cos(IN)) * x_p + \
#             (-np.sin(OM) * np.sin(W) + np.cos(OM) * np.cos(W) * np.cos(IN)) * y_p
#         z = (np.sin(W) * np.sin(IN)) * x_p + \
#             (np.cos(W) * np.sin(IN)) * y_p
        
#         positions.append([x, y, z])

#     return positions


# mars = plot_data(mars_ec, mars_om, mars_a, mars_in, mars_w)
# mars = np.array(mars)

# earth = plot_data(earth_ec, earth_om, earth_a, earth_in, earth_w)
# earth = np.array(earth)

# # Create a PyVista plotter
# plotter = pv.Plotter()

# # Plot the Mars orbit
# mars_polydata = pv.PolyData(mars)
# plotter.add_mesh(mars_polydata, color='red', point_size=2, label='Mars')

# # Plot the Earth orbit
# earth_polydata = pv.PolyData(earth)
# plotter.add_mesh(earth_polydata, color='blue', point_size=2, label='Earth')

# # Create a sphere to represent the Sun
# sun_center = np.array([0, 0, 0])
# sun_radius = 200000  # Adjust the radius as needed
# sun_sphere = pv.Sphere(radius=sun_radius, center=sun_center)
# plotter.add_mesh(sun_sphere, color='yellow', label='Sun')

# # Set the camera position and orientation
# plotter.camera_position = 'iso'
# plotter.camera.zoom(0.8)

# # Set the background color
# # plotter.set_background('black')

# # Remove the axis and measurements
# plotter.hide_axes()
# # plotter.hide_axes_planes()

# # Add a legend
# plotter.add_legend(bcolor=(0, 0, 0), border=False, size=[0.15, 0.15], face=None)

# # Display the plot
# plotter.show()


# Plot the orbit in 3D
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(mars[:, 0], mars[:, 1], mars[:, 2], 'r-', label='mars')
# ax.plot(earth[:, 0], earth[:, 1], earth[:, 2], 'b-', label='earth')
# ax.plot([0], [0], [0], 'ro', markersize=10, label='sun')  # Plot the Sun at the origin
# ax.set_xlabel('X (km)')
# ax.set_ylabel('Y (km)')
# ax.set_zlabel('Z (km)')
# ax.set_title('Orbit of the Body in 3D')
# ax.grid(True)
# ax.set_zlim(-2e6, 2e6)
# plt.legend()
# plt.show()

# import re
# import pandas as pd

# data = []
# import re
# import pandas as pd


# with open('./src/data/earth.json') as file:
#     file_contents = file.read()

# split_data = file_contents.split('*******************************************************************************')

# global data

# for parsing in split_data:
#     if 'm/s^2' in parsing:
#         lines = parsing.split('\\n')
#         data = [line.split('\\') for line in lines if 'm/s^2' in line]
#         print(data)
        # grav = float([splitting[0].split('=') for splitting in data][-1][-1])

    # print(lines)
    # if 'Equ. gravity' in lines[0]:
    #     print(lines)
    # print(lines)







# for parsing in split_data:
#     if '$$SOE' in parsing:
#         lines = parsing.split('\\n')
#         data = [line.split(',')[2:-1] for line in lines if len(line.split(',')) > 1]

# columns = ['EC', 'QR', 'IN', 'OM', 'W', 'TP', 'N', 'MA', 'TA', 'A', 'AD', 'PR']



# df = pd.DataFrame(data, columns=columns)
# print(df)


# print(response[0]["lat"])
# print(response[0]["lon"])

# import pandas as pd
# import numpy as np

# df = pd.read_csv('src/worldcities.csv')

# country = 'united states'
# country = country.title()

# city = 'ogden'
# city = city.title()

# mask_country = np.column_stack([df['country'].str.contains(country, na=False) for col in df])
# mask_city = np.column_stack([df['city'].str.contains(city, na=False) for col in df])

# new_city = ""

# try:
#     lat = df.loc[mask_city.any(axis=1)].iloc[0].lat
#     lng = df.loc[mask_city.any(axis=1)].iloc[0].lng
#     print(np.deg2rad(lng))
#     print(f"a {lat} {lng}")
# except:
#     new_city = df.loc[mask_country.any(axis=1)].iloc[0].city
#     print(f"b {new_city}")
#     if new_city:
#         print(f"Couldn't locate {city}, instead using: {new_city}")
#     else:
#         print(f"No city in {country} could be found. Try again.")

# rplanet = 6378000
# h = rplanet + 600000
# print(h)

# x = h * np.cos(np.deg2rad(lng))
# y = h * np.sin(np.deg2rad(lng))
# z = h * np.sin(np.deg2rad(lat))

# print(x, y, z)
# print(df.iloc[0].city)


# plotter = pv.Plotter(window_size=[2100,1400])
# target = pv.PolyData([sx, sy, sz])
# launch = pv.PolyData([x0, y0, z0])

# orbit_points = np.column_stack((a, b, c))
# plotter.add_mesh(orbit_points, color='blue', label='orbit', style='wireframe', point_size=2)

# plotter.add_mesh(target, color='green', point_size=15, render_points_as_spheres=True, label='Target')     
# plotter.add_mesh(launch, color='red', point_size=15, render_points_as_spheres=True, label='Launch')     
# Earth = examples.planets.load_earth(radius=6371000)
# earth_texture = examples.load_globe_texture()
# plotter.add_mesh(Earth, texture=earth_texture)

# _ = plotter.add_axes(
#     line_width=5,
#     cone_radius=0.3,
#     shaft_length=0.7,
#     tip_length=0.3,
#     ambient=0.5,
#     label_size=(0.4, 0.16),
# )

# plotter.show()
import numpy as np

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(25, 14))
print(np.ndim(ax))
# print(ax[0][0])
# import pyvista as pv
# from pyvista import examples

# plotter = pv.Plotter(window_size=[2100,1400])
# plotter.set_background('black')

# Earth = examples.planets.load_earth(radius=6378.1)
# earth_texture = examples.load_globe_texture()

# plotter.add_mesh(Earth, texture=earth_texture, opacity=0.5)
# plotter.show()

# x, y, z = geo_to_cart(country, city, r, False)
    # print(f'Target {city}=====\nX:{x},\nY:{y},\nZ:{z}')

    # plotter = pv.Plotter(window_size=[2100,1400])
    # point = pv.PolyData([x0, y0, z0])
    # target = pv.PolyData([x, y, z])

    # plotter.add_mesh(point, color='red', point_size=15, render_points_as_spheres=True, label='Launch')
    # plotter.add_mesh(target, color='green', point_size=15, render_points_as_spheres=True, label='Target')

    # plotter.add_legend(size=(0.1, 0.1), face=None, bcolor='white')

    # Earth = examples.planets.load_earth(radius=6371000)
    # earth_texture = examples.load_globe_texture()
    # plotter.add_mesh(Earth, texture=earth_texture)

    # _ = plotter.add_axes(
    #     color='black',
    #     line_width=5,
    #     cone_radius=0.3,
    #     shaft_length=0.7,
    #     tip_length=0.3,
    #     ambient=0.5,
    #     label_size=(0.4, 0.16),
    # )

    # plotter.show()


    #=====================================================================
    # sx, sy, sz = geo_to_cart('united states', 'salt lake city', r, False)

    # phi = np.arctan(sy/sx)

    # xt = r * np.cos(phi)
    # yt = r * np.sin(phi) 

    # x0 = yt
    # y0 = -xt
    # z0 = 0.0

    # theta = np.linspace(0, 2 * np.pi, 10000)

    # a = np.array([np.cos(the)*x0 + np.sin(the)*sx for the in theta])
    # b = np.array([np.cos(the)*y0 + np.sin(the)*sy for the in theta])
    # c = np.array([np.cos(the)*z0 + np.sin(the)*sz for the in theta])

    # x = a[1] - a[0]
    # y = b[1] - b[0]
    # z = c[1] - c[0]

    # v = np.sqrt(mplanet*G/r)

    # r0 = np.sqrt(x**2 + y**2 + z**2)
    # phi = np.arctan2(y, x)
    # theta = np.arccos(z/r0)
    
    # xv0 = v * np.cos(phi) * np.sin(theta) 
    # yv0 = v * np.sin(phi) * np.sin(theta) 
    # zv0 = v * np.cos(theta) 

    # print(f'Target V: {np.sqrt(mplanet*G/r)}')

    # print(f'Current V: {np.sqrt(xv0**2 + yv0**2 + zv0**2)}')

    # test = SateliteOrbit(x0, y0, z0, xv0, yv0, zv0, 1600000, 0.6, G, mplanet, rplanet, 1)
    # stateout = test._solve()
    # print(stateout)
    # test._plot_orbit(stateout)
    #======================================================================


    # def geo_to_cart(country, city, height, output):

    # df = pd.read_csv('src/worldcities.csv')

    # country = country.title()
    # city = city.title()

    # mask_country = np.column_stack([df['country'].str.contains(country, na=False) for col in df])
    # mask_city = np.column_stack([df['city'].str.contains(city, na=False) for col in df])

    # lat = 0
    # lng = 0

    # try:
    #     city = df.loc[mask_city.any(axis=1)].iloc[0]
    #     lat = np.deg2rad(city.lat)
    #     lng = np.deg2rad(city.lng)

    #     if output == True:
    #         print(f'Located {city.city}, {city.country}')
    #         print(f'Lat: {city.lat}, Lng:{city.lng}')
    # except:
    #     try:
    #         new_city = df.loc[mask_country.any(axis=1)].iloc[0]
    #         lat = np.deg2rad(new_city.lat)
    #         lng = np.deg2rad(new_city.lng)

    #         if output == True:
    #             print(f'Could not locate {city}, instead using: {new_city.city}, {country}')
    #             print(f'Lat: {lat}, Lng:{lng}')
    #     except:
    #         print(f'No city in {country} could be found. Try again.')
    #         quit

    # x = height * np.cos(lng)*np.cos(lat)
    # y = height * np.sin(lng)*np.cos(lat)
    # z = height * np.sin(lat)

    # return -x, -y, z


    # assert isinstance(launch_pos[0], float), 'x pos must be float (if dealing with ints type cast or add .0)'
    #     self.x0 = launch_pos[0]
    #     assert isinstance(launch_pos[1], float), 'y pos must be float (if dealing with ints type cast or add .0)'
    #     self.y0 = launch_pos[0]
    #     assert isinstance(launch_pos[2], float), 'z pos must be float (if dealing with ints type cast or add .0)'
    #     self.z0 = launch_pos[0]

    #     assert isinstance(launch_pos[3], float), 'xdot must be float (if dealing with ints type cast or add .0)'
    #     self.xv0 = launch_pos[3]
    #     assert isinstance(launch_pos[4], float), 'ydot must be float (if dealing with ints type cast or add .0)'
    #     self.yv0 = launch_pos[4]
    #     assert isinstance(launch_pos[5], float), 'zdot must be float (if dealing with ints type cast or add .0)'
    #     self.zv0 = launch_pos[5]