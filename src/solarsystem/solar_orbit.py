""" 
    Utilizes JPL Horizons data to convert Osculating Orbital Elements 
    to cartestian coordinates. Calculates orbital path using just first
    step of osculating data. 
"""
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import constants as c

def osc_to_cart(df):

    array = df.iloc[0].values[1:]
    EC = float(array[0])
    IN = np.radians(float(array[2]))
    OM = np.radians(float(array[3]))
    W = np.radians(float(array[4]))
    A = float(array[9])

    positions = []

    true_anomalies = np.linspace(0, 2*np.pi, 1000)

    for ta in true_anomalies:
        # Calculate the distance from the Sun to the body
        r = self.A * (1 - self.EC**2) / (1 +self.EC * np.cos(ta))
        
        # Calculate the position in the perifocal coordinate system
        x_p = r * np.cos(ta)
        y_p = r * np.sin(ta)
        
        x = (np.cos(OM) * np.cos(W) - np.sin(OM) * np.sin(W) * np.cos(IN)) * x_p + \
            (-np.cos(OM) * np.sin(W) - np.sin(OM) * np.cos(W) * np.cos(IN)) * y_p
        y = (np.sin(OM) * np.cos(W) + np.cos(OM) * np.sin(W) * np.cos(IN)) * x_p + \
            (-np.sin(OM) * np.sin(W) + np.cos(OM) * np.cos(W) * np.cos(IN)) * y_p
        z = (np.sin(W) * np.sin(IN)) * x_p + \
            (np.cos(W) * np.sin(IN)) * y_p
        
        positions.append([x, y, z])


    return positions





    