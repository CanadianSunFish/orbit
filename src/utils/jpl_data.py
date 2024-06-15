""" 
    Gets and saves JPL Horizons data based on NAIF integer ID codes, 
    target center, and start and end date. Also provides a list of
    all NAIF integer ID codes.
"""
import io
import os
import json
import pprint
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from utils import data

url = 'https://ssd.jpl.nasa.gov/api/horizons.api'

def id_list():
    print("All available NAIF integer IDs:")
    pprint.pp(data.naif_id_dict)

class JPLData():

    def __init__(
        self, 
        target_id: int,
        data: str,
        center_id: Optional[str] = "500@10",
        start: Optional[str] = '2024-01-01',
        end: Optional[str] = datetime.today().strftime('%Y-%m-%d'),
        step: Optional[str] = '1d'
    ) -> None:
        """Construct `JPLData`
        
            Args:
                target_id: JPL id of astronomical body (i.e 499 - "Mars")
                center_id: JPL id for coordinate origin of astronomical body data.
                start: Start date formatted in '%Y-%m-%d.
                end: Start date formatted in '%Y-%m-%d.
                step: Step size in form '1d, '5d', '1mo', '1yr' etc.

        """

        assert isinstance(target_id, int), 'target id must be an int'
        assert target_id in data.naif_id_dict, id_list()
        self.target_id = target_id

        self.string_id = data.naif_id_dict.get(self.target_id)

        assert isinstance(data, str), 'datatype must be str'
        assert data in data.jpl_api, f'datatype must be one of {data.jpl_api}'
        self.data = data.capitalize()

        assert isinstance(center_id, str), 'center id must be str'
        self.center_id = center_id

        assert isinstance(start, str), 'start must be a str'
        assert isinstance(end, str), 'end must be a str'

        start = str(datetime.strptime(start, '%Y-%m-%d').date())
        end = str(datetime.strptime(end, '%Y-%m-%d').date())

        assert start < end, 'start date must come before end date'

        self.start = start
        self.end = end

        assert isinstance(step, str), 'step size must be string'
        self.step = step

        self.params = {
            'format': 'json',
            'COMMAND': str(self.target_id),  
            'OBJ_DATA': 'YES',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': data,
            'CENTER': center_id,  
            'START_TIME': start,
            'STOP_TIME': end,
            'STEP_SIZE': step,
            'CSV_FORMAT': 'YES'
        }

    def __str__(self):
        return f'Celestial Body: {self.string_id.title()}\nNaif ID: {self.target_id}\nStart date: {self.start}\nEnd date: {self.end}'

    def get_data(self) -> None:
        response = requests.get(url, params=self.params)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        Path(f'{data_dir}').mkdir(parents=True, exist_ok=True)

        if response.status_code == 200:
            data = response.text

            with open(f'{data_dir}/{self.string_id.lower()}.json', 'w') as file:
                json.dump(data, file, indent=4)
            
            self.data_path = f'{data_dir}/{self.string_id.lower()}.json'
            if __name__ == "__main__":
                print(f'{self.string_id.title()} data saved at {data_dir}/{self.string_id.lower()}.json')

        else:
            print('Error:', response.status_code)


class JPLParse():

    def __init__(
        self,
        path: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        data_args: Optional[JPLData] = None
    ) -> None:

        self.data = None
        self.df = None
        
        if data_args is None:
            assert path is not None, "path must be specified"
            self.path = path

            assert start is not None, "start date must be specified"
            assert end is not None, "end date must be specified"
            start = datetime.strptime(start, '%Y-%m-%d').date()
            end = datetime.strptime(end, '%Y-%m-%d').date()

            assert start < end, "start date must come before end date"
            
            self.date_columns = pd.date_range(start, end, freq='d')
        else:
            assert isinstance(data_args, JPLData), "arg must be instance of JPLData class"
            self.data_args = data_args

            self.date_columns = pd.date_range(self.data_args.start, self.data_args.end, freq='d')

            assert self.data_args.data_path is not None, "must call JPLData._get_data first"
            self.path = self.data_args.data_path


    def _parse(self):
        try:
            with open(self.path) as file:
                file_contents = file.read()
        except FileNotFoundError:
            print("data path invalid, try copying the path provided when calling JPLData._get_data()")

        split_data = file_contents.split('*******************************************************************************')

        for parsing in split_data:
            if '$$SOE' in parsing:
                lines = parsing.split('\\n')
                self.data = [line.split(',')[2:-1] for line in lines if len(line.split(',')) > 1]

        if len(self.data[0]) != len(c.columns()):
            raise Exception("something went wrong when parsing the data, check to see if data exists")

        if len(self.data) != len(self.date_columns):
            raise Exception("date args do not")

        self.df = pd.DataFrame(self.data, columns=data.columns())
        self.df.insert(0, "Date", self.date_columns, True)


if __name__ == "__main__":

    test = JPLData(299, 'VECTOR', start='2024-04-01', end='2024-04-10')
    print(test)
    test.get_data()

