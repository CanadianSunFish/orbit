import numpy as np
import pandas as pd

from typing import Optional

class LaunchPosition():

    def __init__(
        self,
        country: str,
        city: Optional[str],
        r: float,
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
        # assert r > 6371000 + 99000, f'orbit radius r must be be above the atmosphere {6371000 + 99000}m, currently {r}m'
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
    
