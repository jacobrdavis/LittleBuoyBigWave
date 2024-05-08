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
    'significant_wave_height',
    'direction',
    'directional_spread',
    'moment_weighted_mean',
    'merge_frequencies',
]

import warnings
from typing import Tuple, Optional, Union

import numpy as np


ACCELERATION_OF_GRAVITY = 9.81  # (m/s^2)
TWO_PI = 2 * np.pi


def mean_square_slope(
    energy_density: np.ndarray,
    frequency: np.ndarray,
) -> Union[float, np.ndarray]:
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
) -> Union[float, np.ndarray]:
    """
    Calculate energy-weighted frequency as the ratio of the first and zeroth
    moments of the one-dimensional frequency spectrum.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        return_as_frequency (bool): if True, return frequency in Hz.

    Returns:
    Energy-weighted period as a
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


def spectral_moment(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """
    Compute the 'nth' spectral moment.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum.
        frequency (np.ndarray): 1-D frequencies.
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        axis (int, optional): Axis to calculate the moment along. Defaults to -1.

    Returns:
    nth moment as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if `energy_density` has more than one dimension.  The shape
            of the returned array is reduced along `axis`.
    """
    frequency_n = frequency ** n
    moment_n = np.trapz(energy_density * frequency_n, x=frequency, axis=axis)
    return moment_n


def moment_weighted_mean(arr, energy_density, frequency, n, axis=-1):
    #TODO:
    moment_n = spectral_moment(energy_density=energy_density,
                               frequency=frequency,
                               n=n,
                               axis=axis)

    weighted_moment_n = spectral_moment(energy_density=energy_density * arr,
                                        frequency=frequency,
                                        n=n,
                                        axis=axis)
    return weighted_moment_n / moment_n


def significant_wave_height(
    energy_density: np.ndarray,
    frequency: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Calculate significant wave height as four times the square root of the
    spectral variance.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).

    Returns:
    Significant wave height as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (n, f).

    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    zeroth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=0,
                                    axis=-1)

    return 4 * np.sqrt(zeroth_moment)


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
    return directional_spread_rad  # np.rad2deg(directional_spread_rad)


def merge_frequencies(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n_merge: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge neighboring frequencies in a spectrum.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        n_merge (int): number of adjacent frequencies to merge.

    Returns:
        Tuple[np.ndarray, np.ndarray]: merged energy density and frequency.

    Example:
    ```
    >>> frequency = np.arange(0, 9, 1)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    >>> energy_density = frequency * 3
    array([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

    >>> merge_frequencies(energy_density, frequency, n_merge=3)
    (array([1., 4., 7.]), array([ 3., 12., 21.]))
    ```
    """
    n_groups = len(frequency) // n_merge
    frequency_merged = _average_n_groups(frequency, n_groups)
    energy_density_merged = np.apply_along_axis(_average_n_groups,
                                                axis=-1,
                                                arr=energy_density,
                                                n_groups=n_groups)
    return energy_density_merged, frequency_merged


def _average_n_groups(arr: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Adapted from Divakar via https://stackoverflow.com/questions/53178018/
    average-of-elements-in-a-subarray.

    Functionally equivalent to (but faster than):
    arr_split = np.array_split(arr, n_groups)
    arr_merged = np.array([group.mean() for group in arr_split])
    """
    n = len(arr)
    m = n // n_groups
    w = np.full(n_groups, m)
    w[:n - m*n_groups] += 1
    sums = np.add.reduceat(arr, np.r_[0, w.cumsum()[:-1]])
    return sums / w
