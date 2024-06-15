"""
    Brings together all elements of OrbitalAnalysis package to provide
    a space programe that simulates rocket launch based on specified
    parameters. 
"""
import time
from utils import jpl_data
from utils import jpl_parse
from orbital_util import *
from datetime import datetime
from utils import constants as c

# import sys
# print(sys.path)
# exit()

today = datetime.today().strftime('%Y-%m-%d')


while(True):

    planets = input(
    """
    Input (integer value of) Planets To Simulate (comma separated)
    1 - Mercury
    2 - Venus
    3 - Earth
    4 - Moon
    5 - Mars
    6 - Jupiter
    7 - Saturn
    8 - Uranus
    9 - Neptune
    10 - Pluto

    Type 'quit' to quit.

    """)
    
    if planets == 'quit':
        break
    
    try:
        planets = planets.split(',')
        planets = [int(i) for i in planets]
    except:
        print("=============")
        print("Pass integers")
        print("=============")
        continue
    naif_id_list = [c.planet_order_dict.get(planet) for planet in planets]
    string_list = [c.naif_id_dict.get(id) for id in naif_id_list]

    data_list = []
    
    for id in naif_id_list:

        a = JPLData(id, 'ELEMENTS')
        a.get_data()
        b = JPLParse(a.data_path, a.start, a.end)
        b._parse()
        d = OrbitalCalculation(b.df)
        d._osculating_to_cartesian()
        data_list.append(d.positions)

    plot(tuple(data_list), string_list)


    # parsed_data = [JPLData(id).get_data() for id in naif_id_list]

    # print(i.data_path for i in parsed_data)
    
    time.sleep(1)