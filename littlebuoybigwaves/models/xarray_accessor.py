"""
Xarray Dataset model accessor and associated methods.
"""

__all__ = [
    "ModelDatasetAccessor",
    "ModelDataArrayAccessor",
]

import types
from typing import Hashable, List, Tuple, Optional
from urllib.error import URLError

import numpy as np
import pandas as pd
import xarray as xr

from littlebuoybigwaves import waves, buoy
from utilities import get_config


#TODO: add default config! try statement?
config = get_config()['littlebuoybigwaves']['model']
config_vars = types.SimpleNamespace(**config['vars'])


@xr.register_dataset_accessor("model")
class ModelDatasetAccessor:
    """ Extend xarray Dataset objects with model-specific functionality. """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None
        self._vars = config_vars

    @property
    def center(self):
        """ Return the geographic center point of this dataset. """
        if self._center is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

    @property
    def vars(self):
        """ Return a SimpleNamespace with this Dataset's variable names. """
        return self._vars

    def fq_dir_spectrum_to_fq_spectrum(self) -> xr.DataArray:
        """
        Integrate frequency-direction spectra over direction to obtain
        scalar frequency spectra and return it as a DataArray.
        """
        fq_dir_spectrum = self._obj[self.vars.frequency_direction_energy_density]
        fq_spectrum = fq_dir_spectrum.integrate(coord=self.vars.direction)
        return fq_spectrum

    # def colocate_with_path_ds(...):  TODO: see example from pywsra
    # def fq_dir_spectrum_to_wn_spectrum(...):

    def mean_square_slope(self) -> xr.DataArray:
        """ Calculate mean square slope and return it as a DataArray. """
        mean_square_slope = xr.apply_ufunc(
            waves.mean_square_slope,
            self._obj[self.vars.frequency_energy_density],
            self._obj[self.vars.frequency],
            input_core_dims=[[self.vars.frequency], [self.vars.frequency]],
            output_core_dims=[[]],
            vectorize=True,
        )
        return mean_square_slope



@xr.register_dataarray_accessor("model")
class ModelDataArrayAccessor:
    """ Extend xarray DataArray objects with model-specific functionality. """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None
