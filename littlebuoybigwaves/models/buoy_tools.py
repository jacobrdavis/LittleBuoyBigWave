#TODO: put this inside of ERA5/GFS tools!
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable
import warnings

def closest_value(x: float,X: Iterable): # lat,lon,latList, lonList):
    """
    Helper function to find index of closest value of x in list X
    
    """
    Xidx = np.argmin(abs(X-x)) 
    return Xidx    



