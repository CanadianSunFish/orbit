#
# Solution getsta.py
#
from __future__ import print_function
from builtins import input

import datetime
import spiceypy as spice

date = datetime.datetime.today()
date = date.strftime("%Y-%m-%dT00:00:00")

# Loading the leap second kernel
LSK = 'src/utils/kernels/lsk/naif0012.tls'
spice.furnsh( LSK )

# Working around J2000 ephemeris time
ephemeris_time = spice.utc2et(date)


# Loading spice kernel
FILENAME = 'src/utils/kernels/spk/de432s.bsp'
spice.furnsh(FILENAME)

# TODO determine if the units are kilometers
mars = spice.spkezr('MARS BARYCENTER', ephemeris_time, 'ECLIPJ2000', 'NONE', 'SUN')[0][:3]
mercury = spice.spkezr('MERCURY BARYCENTER', ephemeris_time, 'ECLIPJ2000', 'NONE', 'SUN')[0][:3]
earth = spice.spkezr('EARTH', ephemeris_time, 'ECLIPJ2000', 'NONE', 'SUN')[0][:3]

from plotting_utils import *

plot = Plotter(
    [mars, mercury, earth], 
    args={
        'multi_array': True, 
        'radius': [data.mars['mean_r'].value, 
                   data.mercury['equ_r'].value,
                   data.earth['mean_r'].value]
    }, 
    labels=('mars', 'mercury', 'earth')
)

plot.solar_system()

def getss():

    # METAKR = '/home/bread/Documents/projects/orbit/src/utils/meta/metakn.tm'
    # spice.furnsh( METAKR )

    

    



    objects = spice.spkobj(FILENAME)

    times = [100 * i for i in range(90)]


    n = 0
    ids, names, tcs_sec, tcs_cal = [],[],[],[]

    for o in objects:

        ids.append(o)
        tc_sec = spice.wnfetd(spice.spkcov(FILENAME, ids[n]), n)
        tc_cal = [spice.timout(f, "YYYY MON DD HR:MN:SC.### (TBD) ::TBD") for f in tc_sec]

        tcs_sec.append(tc_sec)
        tcs_cal.append(tc_cal)


# if __name__ == '__main__':
#     getss()