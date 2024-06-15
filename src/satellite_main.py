from satellite.sat_orbit import *
from utils import constants as c

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
orbit._plot_orbit(state_initial, target)
orbit._plot_states()