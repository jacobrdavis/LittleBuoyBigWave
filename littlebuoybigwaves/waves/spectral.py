"""
Spectral water wave functions.
"""

# TODO:
# - unit tests
# - rename/unify functions and variables


__all__ = [
    'mean_square_slope',
    'energy_period',
    'spectral_moment',
    'sig_wave_height',
    'direction',
    'directional_spread',
    'moment_weighted_mean',
]

import warnings
from typing import Tuple, Optional

import numpy as np


ACCELERATION_OF_GRAVITY = 9.81  # (m/s^2)
TWO_PI = 2 * np.pi


def mean_square_slope(
    energy_density: np.ndarray,
    frequency: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Calculate spectral mean square slope as the fourth moment of the one-
    dimensional frequency spectrum.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (n, f).
    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    fourth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=4,
                                    axis=-1)
    return (TWO_PI**4 * fourth_moment) / (ACCELERATION_OF_GRAVITY**2)


def energy_period(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    return_as_frequency: bool = False
) -> Tuple[float, np.ndarray]:
    """
    Calculate energy-weighted frequency as the ratio of the first and zeroth
    moments of the one-dimensional frequency spectrum.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        return_as_frequency (bool): if True, return frequency in Hz.

    Returns:
    Energy period as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (n, f).

    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    # Ratio of the 1st and 0th moments is equilvaent to 0th moment-
    # weighted frequency.
    energy_frequency = moment_weighted_mean(arr=frequency,
                                            energy_density=energy_density,
                                            frequency=frequency,
                                            n=0)

    if return_as_frequency:
        return energy_frequency
    else:
        return energy_frequency**(-1)


def spectral_moment(energy_density, frequency, n, axis=-1):
    """
    Compute the 'nth' spectral moment.
    """
    frequency_n = frequency ** n
    moment_n = np.trapz(energy_density * frequency_n, x=frequency, axis=axis)
    return moment_n


def moment_weighted_mean(arr, energy_density, frequency, n, axis=-1):
    moment_n = spectral_moment(energy_density=energy_density,
                               frequency=frequency,
                               n=n,
                               axis=axis)

    weighted_moment_n = spectral_moment(energy_density=energy_density * arr,
                                        frequency=frequency,
                                        n=n,
                                        axis=axis)
    return weighted_moment_n / moment_n


def sig_wave_height(energy, freq):
    """
    Function to compute significant wave height
    
    Input:
        - energy, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - freq, input array of frequencies ([n,1] arr OR [n,m] ndarr)
       
    Output:
        - Hs, significant wave height ([1,] float)
            * if energy is empty or invalid, Hs is assigned a NaN

    Example:
    
        Hs = sig_wave_height(energy, freq)

    """
    if hasattr(energy, '__len__') and (not isinstance(energy, str)):
        # recover variance from spectrum:
        m0 = spectral_moment(energy, freq, n=0) 

        # standard deviation:
        stDev = np.sqrt(m0)  

        # sig wave height:   
        Hs  =  4*stDev
      
    else:
        Hs = np.NaN
    return Hs

#TODO: option to use a2 b2?
def direction(a1, b1):
    """TODO: from Spotter Technical Reference Manual"""
    return (270 - np.rad2deg(np.arctan2(b1, a1))) % 360

#TODO: option to use a2 b2?
def directional_spread(a1, b1):
    """
    Calculate directional spreading  by frequency bin using the lowest-
    order directional moments.

    Args:
        a1 (np.ndarray): normalized spectral directional moment (+E)
        b1 (np.ndarray): normalized spectral directional moment (+N)

    Returns:
       np.ndarray: directional spread in radians.
    """
    directional_spread_rad = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
    return directional_spread_rad #np.rad2deg(directional_spread_rad)


#%% testing
def main():
    energy = [0.1, 0.5, 10, 0.5]
    freq = [0.05, 0.1, 0.25, 0.5]
    freq_range = [0.05,0.25]
    # freq_range = 'total'
    mss = mean_square_slope(energy, freq, norm='frequency', freq_range=freq_range)
    print(mss)

if __name__=='__main__':
    main()
#%%

 