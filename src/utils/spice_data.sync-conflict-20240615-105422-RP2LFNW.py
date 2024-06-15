#
# Solution getsta.py
#
from __future__ import print_function
from builtins import input

import spiceypy as spice

def getss():

    METAKR = '/home/bread/Documents/projects/orbit/src/utils/meta/metakn.tm'
    FILENAME = 'src/utils/kernels/spk/de432s.bsp'
    LSK = 'src/utils/kernels/lsk/naif0012.tls'

    spice.furnsh( METAKR )

    spice.furnsh( LSK )

    spice.furnsh(FILENAME)


    objects = spice.spkobj(FILENAME)

    print(spice.spkezr('MARS BARYCENTER', 20000.0, 'ECLIPJ2000', "NONE", 'SUN'))

    n = 0
    ids, names, tcs_sec, tcs_cal = [],[],[],[]

    for o in objects:


        ids.append(o)
        tc_sec = spice.wnfetd(spice.spkcov(FILENAME, ids[n]), n)
        tc_cal = [spice.timout(f, "YYYY MON DD HR:MN:SC.### (TBD) ::TBD") for f in tc_sec]

        tcs_sec.append(tc_sec)
        tcs_cal.append(tc_cal)

    # print(tcs_cal)


if __name__ == '__main__':
    getss()