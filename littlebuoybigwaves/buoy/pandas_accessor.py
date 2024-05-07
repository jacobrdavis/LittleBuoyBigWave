"""
Core module for LittleBuoyBigWaves Pandas interface.  Contains Dataframe
accessors and associated methods.
"""


# TODO:
# - Many methods can be vectorized if all frequencies have the same shape...
#   Might consider adding a check for uniform frequency arrays and subsequent
#   pathways in methods.
# - Create default namespace that can be intersected with config namespace
# Default config cols file + read local config from current working directory.  Print if no local config found and intersect/use default cols

__all__ = [
    "BuoyDataFrameAccessor",
]

import types
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from littlebuoybigwaves import waves, buoy
from utilities import get_config


#TODO: add default config!  try catch?
config = get_config()['littlebuoybigwaves']
config_idx = types.SimpleNamespace(**config['idxs'])
config_cols = types.SimpleNamespace(**config['cols'])


@pd.api.extensions.register_dataframe_accessor("buoy")
class BuoyDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._cols = config_cols  # TODO: probably need a method to update and rename
        self._idxs = config_idx
        # self._spectral_variables = None # TODO: do not want to cache

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        #TODO: need to add col using SimpleNameSpace
        if "latitude" not in obj.columns or "longitude" not in obj.columns:
            raise AttributeError("Must have 'latitude' and 'longitude'.")
        #TODO: validate that the index is a datetime index, at least!

    @property
    def center(self):
        """ Return the geographic center point of this DataFrame. """
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

    @property
    def cols(self):
        """ Return a SimpleNamespace with this DataFrame's column names. """
        return self._cols

    @property
    def idxs(self):
        """ Return a SimpleNamespace with this DataFrame's indice names. """
        return self._idxs

    @property
    def spectral_variables(self) -> List:
        """ Return a list of spectral variables in the DataFrame. """
        # Apply np.size element-wise to generate a DataFrame of sizes
        size_df = self._obj.applymap(np.size, na_action='ignore')

        # Compare each column in size_df to the frequency column and return
        # only the matching columns, which should be spectral.
        is_spectral = size_df.apply(
            lambda col: size_df[self.cols.frequency].equals(col)
        )
        return is_spectral.index[is_spectral].to_list()

    def to_xarray(self) -> xr.Dataset:
        """ Return this DataFrame as an Xarray Dataset. """
        # Bulk (single value) and spectral columns must be handled separately
        # since `.to_xarray()` does not convert elements containing arrays.
        drifter_bulk_ds = (self._obj
                           .drop(columns=self.spectral_variables)
                           .to_xarray())

        drifter_spectral_ds = (self._obj
                               .loc[:, self.spectral_variables]
                               .explode(self.spectral_variables)
                               .set_index(self.cols.frequency, append=True)
                               .to_xarray())

        drifter_ds = xr.merge([drifter_bulk_ds, drifter_spectral_ds])
        drifter_ds[self.idxs.time] = pd.DatetimeIndex(drifter_ds[self.idxs.time].values)
        return drifter_ds

    def frequency_to_wavenumber(self, **kwargs) -> pd.Series:
        """ Convert frequency to wavenumber and return it as a Series. """
        # If depth data is present, use the full relationship. Otherwise, only
        # the deep water relationship can be used.
        if self.cols.depth in self._obj.columns:
            wavenumber = self._obj.apply(
                lambda df: waves.dispersion(
                    df[self.cols.frequency],
                    np.array([df[self.cols.depth]]),
                    **kwargs
                ),
                axis=1,
            )
        else:
            wavenumber = self._obj.apply(
                lambda df: waves.deep_water_dispersion(
                    df[self.cols.frequency],
                    **kwargs,
                ),
                axis=1,
            )
        return wavenumber

    def mean_square_slope(self, **kwargs) -> pd.Series:
        """ Calculate mean square slope and return it as a Series. """
        mean_square_slope = self._obj.apply(
                lambda df: waves.mean_square_slope(
                    energy_density=df[self.cols.energy_density],
                    frequency=df[self.cols.frequency],
                    **kwargs,
                ),
                axis=1,
            )
        return mean_square_slope

    def energy_period(self, **kwargs) -> pd.Series:
        """ Calculate energy-weighted period and return it as a Series. """
        energy_period = self._obj.apply(
                lambda df: waves.energy_period(
                    energy_density=df[self.cols.energy_density],
                    frequency=df[self.cols.frequency],
                    **kwargs,
                ),
                axis=1,
            )
        return energy_period

    def wave_direction(self, **kwargs) -> pd.Series:
        """
        Calculate wave direction per frequency and return it as a Series.
        """
        direction = self._obj.apply(
                lambda df: waves.direction(
                    df[self.cols.a1],
                    df[self.cols.b1],
                    **kwargs,
                ),
                axis=1,
            )
        return direction

    def wave_directional_spread(self, **kwargs) -> pd.Series:
        """
        Calculate wave directional spread per frequency and return it as a
        Series.
        """
        directional_spread = self._obj.apply(
                lambda df: waves.directional_spread(
                    df[self.cols.a1],
                    df[self.cols.b1],
                    **kwargs,
                ),
                axis=1,
            )
        return directional_spread

    def drift_speed_and_direction(self) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate drift speed and direction and return each as as Series.
        """
        #TODO: need to group by id for correct results
        drift_speed_mps, drift_dir_deg = buoy.drift_speed_and_direction(
            longitude=self._obj[self.cols.longitude],
            latitude=self._obj[self.cols.latitude],
            time=self._obj.index.get_level_values(level=self.idxs.time),
            append=True,
        )
        return drift_speed_mps, drift_dir_deg

    def doppler_correct(self) -> pd.DataFrame:
        """ Doppler correct frequencies and return a new DataFrame."""
        #TODO: need to group by id
        new_df = self._obj.copy(deep=False)  # TODO: confirm this is right
        # Calculate any missing columns.
        if self.cols.wavenumber not in self._obj.columns:
            new_df[self.cols.wavenumber] = new_df.buoy.frequency_to_wavenumber()
        if self.cols.direction not in self._obj.columns:
            new_df[self.cols.direction] = new_df.buoy.wave_direction()
        if self.cols.drift_speed not in self._obj.columns:
            new_df[self.cols.drift_speed], new_df[self.cols.drift_direction] \
                                    = new_df.buoy.drift_speed_and_direction()

        # Apply the Doppler correction to each frequency array. Results can be
        # added directly to the DataFrame copy using `result_type='expand'`.
        #TODO: can be vectorized if all frequencies have the same shape...
        new_cols = [self.cols.absolute_frequency,
                    self.cols.u_dot_k,
                    self.cols.wave_drift_alignment]
        new_df[new_cols] = new_df.apply(
            lambda df: buoy.doppler_correct(
                drift_direction_going=np.array([df[self.cols.drift_direction]]),
                wave_direction_coming=df[self.cols.direction],
                drift_speed=np.array([df[self.cols.drift_speed]]),
                intrinsic_frequency=df[self.cols.frequency],
                wavenumber=df[self.cols.wavenumber],
            ),
            axis=1,
            result_type='expand',
        )
        return new_df

    def doppler_correct_mean_square_slope(self) -> pd.DataFrame:
        """ Doppler correct mean square slope and return a new DataFrame."""
        new_df = self._obj.copy(deep=False)

        # Calculate any missing columns.
        if self.cols.wavenumber not in self._obj.columns:
            new_df[self.cols.wavenumber] = new_df.buoy.frequency_to_wavenumber()
        if self.cols.direction not in self._obj.columns:
            new_df[self.cols.direction] = new_df.buoy.wave_direction()
        if self.cols.drift_speed not in self._obj.columns:
            new_df[self.cols.drift_speed], new_df[self.cols.drift_direction] \
                                    = new_df.buoy.drift_speed_and_direction()

        new_df['mean_square_slope_absolute'] = new_df.apply(  #TODO: use .cols
            lambda df: buoy.doppler_correct_mean_square_slope(
                drift_direction_going=np.array([df[self.cols.drift_direction]]),
                wave_direction_coming=df[self.cols.direction],
                drift_speed=np.array([df[self.cols.drift_speed]]),
                frequency=df[self.cols.frequency],
                energy_density=df[self.cols.energy_density],
            ),
            axis=1,
        )
        return new_df


    #TODO: map plot methods
    def plot(self):
        # plot this array's data on a map, e.g., using Cartopy
        pass


# @pd.api.extensions.register_dataframe_accessor("buoy")
# class BuoySeriesAccessor:
#     def __init__(self, pandas_obj):
#         self._validate(pandas_obj)
#         self._obj = pandas_obj
#         self._idxs = default_idxs

    #TODO:
    # @property
    # def is_spectral(self) -> Bool:
    #     """ Return a list of spectral variables in the DataFrame. """
    #     # Apply np.size element-wise to generate a DataFrame of sizes
    #     size_df = self._obj.applymap(np.size, na_action='ignore')

    #     # Compare each column in size_df to the frequency column and return
    #     # only the matching columns, which should be spectral.
    #     is_spectral = size_df.apply(
    #         lambda col: size_df[self.cols.frequency].equals(col)
    #     )
    #     return is_spectral.index[is_spectral].to_list()

    # def moment_weighted_mean(self):

    #     waves.moment_weighted_mean(
    #         self._obj,
    #         energy_density, frequency, n)
