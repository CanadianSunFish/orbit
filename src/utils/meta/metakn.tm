KPL/MK

   This is the meta-kernel used in the solution of the
   "Obtaining Target States and Positions" task in the
   Remote Sensing Hands On Lesson.

   The names and contents of the kernels referenced by this
   meta-kernel are as follows:

   File name                   Contents
   --------------------------  -----------------------------
   naif0012.tls                Generic LSK
   de432s.bsp                  SOLAR SYSTEM EPHEMERIS DEC 14 1949 JAN 02 2050
   mar097.bsp                  MARS PHOBOS DEIMOS EPHEMERIS JAN 04 1900 JAN 03 2100

   \begindata
   KERNELS_TO_LOAD = ( '/home/bread/Documents/projects/orbit/src/utils/kernels/lsk/naif0012.tls',
                       '/home/bread/Documents/projects/orbit/src/utils/kernels/spk/de432s.bsp',
                       '/home/bread/Documents/projects/orbit/src/utils/kernels/spk/mar097.bsp',
   \begintext