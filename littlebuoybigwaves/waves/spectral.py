"""
Spectral water wave functions.
"""

# TODO:
# - unit tests
# - rename/unify functions and variables


__all__ = [
    'mean_square_slope',
    'wavenumber_mean_square_slope',
    'energy_period',
    'spectral_moment',
    'significant_wave_height',
    'direction',
    'directional_spread',
    'moment_weighted_mean',
    'merge_frequencies',
    'fq_energy_to_wn_energy',
]

import warnings
from typing import Tuple, Optional, Union

import numpy as np

from littlebuoybigwaves import waves

ACCELERATION_OF_GRAVITY = 9.81  # (m/s^2)
TWO_PI = 2 * np.pi


def mean_square_slope(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate spectral mean square slope as the fourth moment of the one-
    dimensional frequency spectrum.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (n, f).
    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    if min_frequency is None:
        min_frequency = frequency.min()

    if max_frequency is None:
        max_frequency = frequency.max()

    # Mask frequencies outside of the specified range.
    frequency_mask = np.logical_and(frequency >= min_frequency,
                                    frequency <= max_frequency)
    frequency = frequency[frequency_mask]
    energy_density = energy_density[..., frequency_mask]

    # Calculate the fourth moment of the energy density spectrum.
    fourth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=4,
                                    axis=-1)
    return (TWO_PI**4 * fourth_moment) / (ACCELERATION_OF_GRAVITY**2)


def wavenumber_mean_square_slope(
    energy_density_wn: np.ndarray,
    wavenumber: np.ndarray
) -> np.ndarray:
    """
    Calculate mean square slope as the second moment of the one-dimensional
    wavenumber spectrum.

    Args:
        energy_density_wn (np.ndarray): 1-D energy density wavenumber spectrum
            with shape (k,) or (n, k).
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (k,).
        np.ndarray: if the shape of `energy_density` is (n, k).
    """
    return np.trapz(y=energy_density_wn * wavenumber**2, x=wavenumber, axis=-1)


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


#TODO: can accept no frequency if n=0 (make it NaN)
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


# def merge_frequencies(
#     energy_density: np.ndarray,
#     frequency: np.ndarray,
#     n_merge: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Merge neighboring frequencies in a spectrum.

#     Args:
#         energy_density (np.ndarray): 1-D energy density frequency spectrum with
#             shape (f,) or (n, f).
#         frequency (np.ndarray): 1-D frequencies with shape (f,).
#         n_merge (int): number of adjacent frequencies to merge.

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: merged energy density and frequency.

#     Example:
#     ```
#     >>> frequency = np.arange(0, 9, 1)
#     array([0, 1, 2, 3, 4, 5, 6, 7, 8])

#     >>> energy_density = frequency * 3
#     array([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

#     >>> merge_frequencies(energy_density, frequency, n_merge=3)
#     (array([1., 4., 7.]), array([ 3., 12., 21.]))
#     ```
#     """
#     n_groups = len(frequency) // n_merge
#     frequency_merged = _average_n_groups(frequency, n_groups)
#     energy_density_merged = np.apply_along_axis(_average_n_groups,
#                                                 axis=-1,
#                                                 arr=energy_density,
#                                                 n_groups=n_groups)
#     return energy_density_merged, frequency_merged


def merge_frequencies(
    spectrum: np.ndarray,
    n_merge: int,
) -> np.ndarray:
    """
    Merge neighboring frequencies in a spectrum.

    Args:
        spectrum (np.ndarray): 1-D frequency spectrum with shape (f,) or (n, f).
        n_merge (int): number of adjacent frequencies to merge.

    Returns:
        Tuple[np.ndarray, np.ndarray]: merged energy density and frequency.

    Example:
    ```
    >>> frequency = np.arange(0, 9, 1)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    >>> energy_density = frequency * 3
    array([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

    >>> merge_frequencies(frequency, n_merge=3)
    array([1., 4., 7.])

    >>> merge_frequencies(energy_density, n_merge=3)
    array([ 3., 12., 21.])
    ```
    """
    n_frequencies = spectrum.shape[-1]
    n_groups = n_frequencies // n_merge
    spectrum_merged = np.apply_along_axis(_average_n_groups,
                                          axis=-1,
                                          arr=spectrum,
                                          n_groups=n_groups)
    return spectrum_merged


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


def fq_energy_to_wn_energy(
    energy_density_fq: np.ndarray,
    frequency: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    var_rtol: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Transform energy density from frequency to wavenumber space.

    Transform energy density, defined in the frequency domain, to energy
    density on a wavenumber domain using the appropriate Jacobians:

    E(k) = E(w) dw/dk

    and

    E(w) = E(k) dk/dw

    where dw/dk is equivalent to the group velocity and dk/dw = 1/(2pi) [1].
    This conversion relies on the (inverse) linear dispersion relationship to
    calculate wavenumbers from the provided frequencies.

    References:
        1. L. H. Holthuijsen (2007) Waves in Oceanic and Coastal Waters,
        Cambridge University Press

    Args:
        energy_density_fq (np.ndarray): 1-D energy density frequency spectrum
            with shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        depth (float, optional): water depth (positive down) with shape (n,).
            Defaults to np.inf.
        var_rtol (float, optional): relative tolerance passed to np.isclose
            to check variance equality. Defaults to 0.02.

    Returns:
        Tuple[np.ndarray, np.ndarray]: energy density in wavenumber domain and
            the corresponding wavenumbers.
    """
    if np.isscalar(depth):
        depth = np.array([depth])

    wavenumber = waves.dispersion(frequency, depth).squeeze()
    dw_dk = waves.intrinsic_group_velocity(wavenumber, frequency, depth)
    df_dw = 1 / (2*np.pi)
    energy_density_wn = energy_density_fq * df_dw * dw_dk
    var_match = check_spectral_variance(energy_density_wn, wavenumber,
                                        energy_density_fq, frequency,
                                        rtol=var_rtol)
    if not var_match:
        raise ValueError('Variance mismatch')
        #TODO: should replace with NaN here
    return energy_density_wn, wavenumber


def check_spectral_variance(
    energy_density_wn: np.ndarray,
    wavenumber: np.ndarray,
    energy_density_fq: np.ndarray,
    frequency: np.ndarray,
    **kwargs
) -> bool:
    """Check for variance equality between wavenumber and frequency spectra.

    Note: Tolerances are specified using the absolute (atol) and relative
    (rtol) tolerance arguments passed to np.isclose via **kwargs.

    Args:
        energy_density_wn (np.ndarray): energy density in wavenumber domain.
        wavenumber (np.ndarray): wavenumber array.
        energy_density_fq (np.ndarray): energy density in frequency domain.
        frequency (np.ndarray): frequency array.
        **kwargs: Additional kwarg arguments are passed to np.isclose.

    Returns:
        bool: True if variance matches within tolerance.
    """
    var_wn = np.trapz(energy_density_wn, x=wavenumber)
    var_fq = np.trapz(energy_density_fq, x=frequency)
    return np.isclose(var_wn, var_fq, **kwargs)