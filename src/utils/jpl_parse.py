"""
    Parses JPL Horizons osculating orbit data from json to pd.Dataframe.. 
"""

import pandas as pd
from typing import Optional
from datetime import datetime
from utils import constants as c

from utils.jpl_data import *

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

        self.df = pd.DataFrame(self.data, columns=c.columns())
        self.df.insert(0, "Date", self.date_columns, True)


if __name__ == "__main__":
    x = 1
