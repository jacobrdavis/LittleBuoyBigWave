"""
Pandas Dataframe buoy accessor and associated methods.
"""


# TODO:
# - Construct idxs by intersecting vars with index names?
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

from littlebuoybigwaves import waves, buoy, utilities

#TODO: add default config!
var_namespace = utilities.get_var_namespace(subset='buoy')

@pd.api.extensions.register_dataframe_accessor("buoy")
class BuoyDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._vars = var_namespace

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
    def vars(self):
        """ Return a SimpleNamespace with this DataFrame's variable names. """
        return self._vars

    @property
    def spectral_variables(self) -> List:
        """ Return a list of spectral variables in the DataFrame. """
        # Apply np.size element-wise to generate a DataFrame of sizes
        size_df = self._obj.applymap(np.size, na_action='ignore')

        # Compare each column in size_df to the frequency column and return
        # only the matching columns, which should be spectral.
        is_spectral = size_df.apply(
            lambda col: size_df[self.vars.frequency].equals(col)
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
                               .set_index(self.vars.frequency, append=True)
                               .to_xarray())

        drifter_ds = xr.merge([drifter_bulk_ds, drifter_spectral_ds])
        drifter_ds[self.vars.time] = pd.DatetimeIndex(drifter_ds[self.vars.time].values)
        return drifter_ds

    def frequency_to_wavenumber(self, **kwargs) -> pd.Series:
        """ Convert frequency to wavenumber and return it as a Series. """
        # If depth data is present, use the full relationship. Otherwise, only
        # the deep water relationship can be used.
        if self.vars.depth in self._obj.columns:
            wavenumber = self._obj.apply(
                lambda df: waves.dispersion(
                    df[self.vars.frequency],
                    np.array([df[self.vars.depth]]),
                    **kwargs
                ),
                axis=1,
            )
        else:
            wavenumber = self._obj.apply(
                lambda df: waves.deep_water_dispersion(
                    df[self.vars.frequency],
                    **kwargs,
                ),
                axis=1,
            )
        return wavenumber

    def mean_square_slope(self, **kwargs) -> pd.Series:
        """ Calculate mean square slope and return it as a Series. """
        mean_square_slope = self._obj.apply(
                lambda df: waves.mean_square_slope(
                    energy_density=df[self.vars.energy_density],
                    frequency=df[self.vars.frequency],
                    **kwargs,
                ),
                axis=1,
            )
        return mean_square_slope

    def energy_period(self, **kwargs) -> pd.Series:
        """ Calculate energy-weighted period and return it as a Series. """
        energy_period = self._obj.apply(
                lambda df: waves.energy_period(
                    energy_density=df[self.vars.energy_density],
                    frequency=df[self.vars.frequency],
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
                    df[self.vars.a1],
                    df[self.vars.b1],
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
                    df[self.vars.a1],
                    df[self.vars.b1],
                    **kwargs,
                ),
                axis=1,
            )
        return directional_spread

    def drift_speed_and_direction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate drift speed and direction and return each as as Series.
        """
        #TODO: need to group by id for correct results
        drift_speed_mps, drift_dir_deg = buoy.drift_speed_and_direction(
            longitude=self._obj[self.vars.longitude],
            latitude=self._obj[self.vars.latitude],
            time=self._obj.index.get_level_values(level=self.vars.time),
            append=True,
        )
        return drift_speed_mps, drift_dir_deg

    def wavenumber_energy_density(self, **kwargs) -> pd.DataFrame:
        """
        Convert frequency domain wave energy density to wavenumber domain
        energy density and return a DataFrame with the energy and wavenumber.
        """
        wavenumber_energy_density = self._obj.apply(
            lambda df: waves.fq_energy_to_wn_energy(
                energy_density_fq=df[self.vars.energy_density],
                frequency=df[self.vars.frequency],
                depth=df[self.vars.depth],
                **kwargs,
            ),
            result_type='expand',
            axis=1,
            #TODO: raw = True,
        )
        #TODO: name columns?
        return wavenumber_energy_density

    def moment_weighted_mean(self, column: str, n: int = 0) -> pd.Series:
        """ Return the nth moment-weighted mean of a column as a Series. """
        moment_weighted_mean_series = self._obj.apply(
            lambda df: waves.moment_weighted_mean(
                arr=df[column],
                energy_density=df[self.vars.energy_density],
                frequency=df[self.vars.frequency],
                n=n,
            ),
            axis=1,
        )
        return moment_weighted_mean_series

    def doppler_correct(self) -> pd.DataFrame:
        """ Doppler correct frequencies and return a new DataFrame."""
        #TODO: need to group by id
        new_df = self._obj.copy(deep=False)  # TODO: confirm this is right
        # Calculate any missing columns.
        if self.vars.wavenumber not in self._obj.columns:
            new_df[self.vars.wavenumber] = new_df.buoy.frequency_to_wavenumber()
        if self.vars.direction not in self._obj.columns:
            new_df[self.vars.direction] = new_df.buoy.wave_direction()
        if self.vars.drift_speed not in self._obj.columns:
            new_df[self.vars.drift_speed], new_df[self.vars.drift_direction] \
                                    = new_df.buoy.drift_speed_and_direction()

        # Apply the Doppler correction to each frequency array. Results can be
        # added directly to the DataFrame copy using `result_type='expand'`.
        #TODO: can be vectorized if all frequencies have the same shape...
        new_cols = [self.vars.absolute_frequency,
                    self.vars.u_dot_k,
                    self.vars.wave_drift_alignment]
        new_df[new_cols] = new_df.apply(
            lambda df: buoy.doppler_correct(
                drift_direction_going=np.array([df[self.vars.drift_direction]]),
                wave_direction_coming=df[self.vars.direction],
                drift_speed=np.array([df[self.vars.drift_speed]]),
                intrinsic_frequency=df[self.vars.frequency],
                wavenumber=df[self.vars.wavenumber],
            ),
            axis=1,
            result_type='expand',
        )
        return new_df

    def doppler_correct_mean_square_slope(self) -> pd.DataFrame:
        """ Doppler correct mean square slope and return a new DataFrame."""
        new_df = self._obj.copy(deep=False)

        # Calculate any missing columns.
        if self.vars.wavenumber not in self._obj.columns:
            new_df[self.vars.wavenumber] = new_df.buoy.frequency_to_wavenumber()
        if self.vars.direction not in self._obj.columns:
            new_df[self.vars.direction] = new_df.buoy.wave_direction()
        if self.vars.drift_speed not in self._obj.columns:
            new_df[self.vars.drift_speed], new_df[self.vars.drift_direction] \
                                    = new_df.buoy.drift_speed_and_direction()

        new_df['mean_square_slope_absolute'] = new_df.apply(  #TODO: use .cols
            lambda df: buoy.doppler_correct_mean_square_slope(
                drift_direction_going=np.array([df[self.vars.drift_direction]]),
                wave_direction_coming=df[self.vars.direction],
                drift_speed=np.array([df[self.vars.drift_speed]]),
                frequency=df[self.vars.frequency],
                energy_density=df[self.vars.energy_density],
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
#         # self._idxs = default_idxs

    #TODO:
    # @property
    # def is_spectral(self) -> Bool:
    #     """ Return a list of spectral variables in the DataFrame. """
    #     # Apply np.size element-wise to generate a DataFrame of sizes
    #     size_df = self._obj.applymap(np.size, na_action='ignore')

    #     # Compare each column in size_df to the frequency column and return
    #     # only the matching columns, which should be spectral.
    #     is_spectral = size_df.apply(
    #         lambda col: size_df[self.vars.frequency].equals(col)
    #     )
    #     return is_spectral.index[is_spectral].to_list()

    # def moment_weighted_mean(self):

    #     waves.moment_weighted_mean(
    #         self._obj,
    #         energy_density, frequency, n)
