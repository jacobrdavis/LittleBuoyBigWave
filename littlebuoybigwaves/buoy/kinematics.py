"""
Kinematics functions.
"""

# TODO:
# - rename/unify functions and variables


__all__ = [
    "drift_speed_and_direction",
    "doppler_shift",  # TODO: remove
    "doppler_correct",  # TODO: remove
    "doppler_adjust",
    "doppler_correct_mean_square_slope",
    "coming_to_going",
    "going_to_coming",
    "frequency_to_angular_frequency",
    "angular_frequency_to_frequency",
    "wave_drift_alignment",
]

from typing import Tuple

import pandas as pd
import numpy as np

from littlebuoybigwaves.geo import haversine_distance

ACCEL_GRAVITY = 9.81  # m/s^2
KMPH_TO_MPS = 0.277778  # m/s / km/hr


def drift_speed_and_direction(
    longitude: np.ndarray,
    latitude: np.ndarray,
    time: pd.DatetimeIndex,
    append: bool = False
) -> Tuple:
    # TODO: docstr

    # Difference the input times to obtain time deltas (in hours).
    time_difference_sec = time[1:] - time[0:-1]
    time_difference_hr = time_difference_sec.seconds / 3600
    time_difference_hr = time_difference_hr.to_numpy()

    # Compute the great circle distance and true bearing between each position.
    dist_km, drift_dir_deg = haversine_distance(longitude=longitude,
                                                latitude=latitude)

    # Calculate drift magnitude; convert from km/hr to m/s.
    drift_speed_kmph = dist_km/time_difference_hr
    drift_speed_mps = drift_speed_kmph * KMPH_TO_MPS

    # If `append` is truthy, append the last value to maintain input size (n,).
    if append:
        drift_speed_mps = _append_last(drift_speed_mps)
        drift_dir_deg = _append_last(drift_dir_deg)

    return drift_speed_mps, drift_dir_deg


def drift_speed_components(drift_speed, drift_dir_deg):
    #TODO: need to define orientation
    #TODO: need to validate
    east_drift_speed = drift_speed * np.sin(np.deg2rad(drift_dir_deg))
    north_drift_speed = drift_speed * np.cos(np.deg2rad(drift_dir_deg))
    return east_drift_speed, north_drift_speed


def doppler_shift(*args, **kwargs):
    """ Alias to renamed function `doppler_correct` """
    return doppler_correct(*args, **kwargs)


# def haversine_distance(longitude, latitude, **kwargs):
#     """ Alias to renamed function `great_circle_pathwise` """
#     return great_circle_pathwise(longitude, latitude, **kwargs)


#TODO: remove (naive implementation without solving for k or Jacobian)
def doppler_correct(
    drift_direction_going: np.ndarray,
    wave_direction_coming: np.ndarray,
    drift_speed: np.ndarray,
    intrinsic_frequency: np.ndarray,
    wavenumber: np.ndarray,
) -> Tuple:
    # TODO: specify shapes...
    # TODO: fix!
    # https://github.com/lcolosi/WaveSpectrum/blob/main/tools/map_omni_dir_spectrum.m
    # Compute drift-wave misalignment.
    wave_direction_going = coming_to_going(wave_direction_coming, modulus=360)
    misalignment_deg = wave_drift_alignment(wave_direction_going, drift_direction_going)

    # Compute the dot product of the drift velocity and wavenumber.
    misalignment_rad = np.deg2rad(misalignment_deg)
    u_dot_k = drift_speed[:, None] * wavenumber * np.cos(misalignment_rad)

    # Adjust the intrinsic frequency by u dot k; note units of rad/s.
    intrinsic_angular_frequency = frequency_to_angular_frequency(intrinsic_frequency)
    absolute_angular_frequency = intrinsic_angular_frequency + u_dot_k
    absolute_frequency = angular_frequency_to_frequency(absolute_angular_frequency)

    return absolute_frequency.squeeze(), u_dot_k.squeeze(), misalignment_deg.squeeze()


def doppler_adjust(
    energy_density_obs: np.ndarray,
    frequency_obs: np.ndarray,
    drift_speed: float,
    drift_direction_going: float,
    wave_direction_coming: np.ndarray,
    frequency_cutoff: float,
    interpolate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Doppler adjust a 1-D spectrum observed on a moving platform to the
    intrinsic reference frame using the omnidirectional solutions described in
    Collins et al. (2017) and Colosi et al. (2023).

    Adapted from the map_omni_dir_spectrum.m source code for:

    Luke Colosi, Nicholas Pizzo, Laurent Grare, Nick Statom, and Luc Lenain.
    "Observations of Surface Gravity Wave Spectra from Moving Platforms."
    Journal of Atmospheric and Oceanic Technology (2023)

    Available at: https://github.com/lcolosi/WaveSpectrum/tree/main

    Args:
        energy_density_obs (np.ndarray): 1-D energy density frequency spectrum
            with shape (f,).
        frequency_obs (np.ndarray): 1-D frequencies in the observed reference
            frame with shape (f,).
        drift_speed (float): platform drift speed.
        drift_direction_going (float): platform drift direction in degrees.
        wave_direction_coming (np.ndarray): wave direction in degrees at each
            frequency with shape (f,).
        frequency_cutoff (float): maximum frequency measurable by the platform.
        interpolate (bool, optional): If True, interpolate the intrinsic energy
            densities onto the observed frequency bins. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: intrinsic energy density and frequency
            arrays with shapes (f,) and (f,).
    """
    # Compute drift-wave misalignment.
    wave_direction_going = coming_to_going(wave_direction_coming, modulus=360)
    misalignment_deg = wave_drift_alignment(wave_direction_going,
                                            np.array([drift_direction_going]))
    misalignment_rad = np.deg2rad(misalignment_deg).squeeze()

    # Compute the Doppler shift speed projection.
    cos_misalignment = np.cos(misalignment_rad)
    projected_speed = drift_speed * cos_misalignment

    # Compute intrinsic frequency handling each frequency by branch.
    frequency_int = np.full(frequency_obs.shape, np.nan)

    # Branch 1: moving against waves
    is_branch_1 = (cos_misalignment < 0) & (drift_speed > 0)
    frequency_int[is_branch_1] = _branch_1(frequency_obs[is_branch_1],
                                           projected_speed[is_branch_1])
    # Branch 2: moving normal to waves
    is_branch_2 = (cos_misalignment == 0) | (drift_speed == 0)
    frequency_int[is_branch_2] = _branch_2(frequency_obs[is_branch_2])

    # Branch 3: moving with waves
    is_branch_3 = (cos_misalignment > 0) & (drift_speed > 0)
    frequency_int[is_branch_3] = _branch_3(frequency_obs[is_branch_3],
                                           projected_speed[is_branch_3],
                                           frequency_cutoff)

    # Approximate Jacobian using central finite differencing and map observed
    # energy density to intrinsic energy density.
    jacobian = np.gradient(frequency_obs, frequency_int)
    energy_density_int = energy_density_obs * jacobian

    #TODO: uniform frequency approach:
    # df_obs = np.unique(np.round(np.diff(frequency_obs), 4))
    # df_int = np.diff(frequency_int)
    # jacobian = df_obs / df_int
    # energy_density_int = np.full(energy_density_obs.shape, np.nan)
    # energy_density_int[0:len(jacobian)] = energy_density_obs[0:len(jacobian)] * jacobian

    # If True, interpolate the intrinsic energy density onto the observed
    # frequency array.
    if interpolate:
        energy_density_int = np.interp(x=frequency_obs,
                                       xp=frequency_int,
                                       fp=energy_density_int)
        frequency_int = frequency_obs.copy()
    else:
        pass

    return energy_density_int, frequency_int


def _frequency_int_solution(frequency_obs, projected_speed):
    """
    Doppler adjust observed frequencies to intrinsic frequencies for a
    platform moving in waves. This is equation (7) in Colosi et al. (2023).
    """
    g = ACCEL_GRAVITY
    discriminant = g**2 - 8*np.pi * g * projected_speed * frequency_obs
    denominator = 4*np.pi * projected_speed
    return (g - np.sqrt(discriminant)) / denominator


def _branch_1(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    against waves.
    """
    return _frequency_int_solution(frequency_obs, projected_speed)


def _branch_2(frequency_obs):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    normal to waves (or not at all).
    """
    return frequency_obs


def _branch_3(frequency_obs, projected_speed, frequency_cutoff):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves.
    """
    # Compute intrinsic frequency at the observed frequency value where
    # df_int(frequency_obs)/df_obs tends towards infinity.
    frequency_int_max = ACCEL_GRAVITY / (4*np.pi * projected_speed)

    # Compute intrinsic frequency when observed frequency = 0.
    frequency_int_zero = ACCEL_GRAVITY / (2*np.pi * projected_speed)

    # Compute intrinsic frequency handling each frequency by case.
    frequency_int = np.full(frequency_obs.shape, np.nan)

    # Case 1: Platform moving slower than energy and crests
    is_case_1 = frequency_int_max > frequency_cutoff
    frequency_int[is_case_1] = _branch_3_case_1(frequency_obs[is_case_1],
                                                projected_speed[is_case_1],
                                                frequency_cutoff)

    # Case 2: Platform moving faster than energy but slower than crests
    is_case_2 = ((frequency_int_max > 0) &
                 (frequency_int_max < frequency_cutoff) &
                 (frequency_int_zero > frequency_cutoff))
    frequency_int[is_case_2] = _branch_3_case_2(frequency_obs[is_case_2],
                                                projected_speed[is_case_2])

    # Case 3: Platform moving faster than energy and crests
    is_case_3 = frequency_int_zero < frequency_cutoff
    frequency_int[is_case_3] = _branch_3_case_3(frequency_obs[is_case_3],
                                                projected_speed[is_case_3])

    return frequency_int


def _branch_3_case_1(frequency_obs, projected_speed, frequency_cutoff):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving slower than the energy and crests.
    """
    # Compute the intrnisic frequency value where it will exceed the cutoff.
    frequency_obs_cutoff = -(2*np.pi * projected_speed * frequency_cutoff**2
                             / ACCEL_GRAVITY) + frequency_cutoff
    above_cutoff = frequency_obs > frequency_obs_cutoff

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


def _branch_3_case_2(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving faster than the energy but slower
    than the crests.
    """
    # Compute observed frequency value where df_int(frequency_obs)/df_obs tends
    # towards infinity.
    frequency_obs_max = ACCEL_GRAVITY / (8*np.pi * projected_speed)
    above_cutoff = frequency_obs > frequency_obs_max

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


def _branch_3_case_3(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving faster than the energy and crests.
    """
    # Compute observed frequency value where df_int(frequency_obs)/df_obs tends
    # towards infinity.
    frequency_obs_max = ACCEL_GRAVITY / (8*np.pi * projected_speed)
    above_cutoff = frequency_obs > frequency_obs_max

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


#TODO: remove
def doppler_correct_mean_square_slope(
    drift_direction_going: np.ndarray,
    wave_direction_coming: np.ndarray,
    drift_speed: np.ndarray,
    frequency,
    energy_density,
) -> np.ndarray:
    #TODO:
    g = 9.81

    # Compute drift-wave misalignment.
    wave_direction_going = coming_to_going(wave_direction_coming, modulus=360)
    misalignment_deg = wave_drift_alignment(wave_direction_going, drift_direction_going)

    misalignment_rad = np.deg2rad(misalignment_deg)
    u_cos_theta = drift_speed[:, None] * np.cos(misalignment_rad)

    mss = 16 * (np.pi * frequency)**4 * energy_density / g**2
    ds1 = 8 * (np.pi * frequency * u_cos_theta)**1 / g**1
    ds2 = 24 * (np.pi * frequency * u_cos_theta)**2 / g**2
    ds3 = 32 * (np.pi * frequency * u_cos_theta)**3 / g**3
    ds4 = 16 * (np.pi * frequency * u_cos_theta)**4 / g**4
    mss_corrected = np.trapz(mss * (1 + ds1 + ds2 + ds3 + ds4), x=frequency)
    return mss_corrected.squeeze()


def wave_drift_alignment(  #TODO: pick one: alignment or misalignment?
    wave_direction_going: np.ndarray,
    drift_direction_going: np.ndarray,
) -> np.ndarray:
    misalignment_full_deg = drift_direction_going[:, None] - wave_direction_going
    misalignment_deg = (misalignment_full_deg + 180) % 360 - 180
    return misalignment_deg


def coming_to_going(coming_from: np.ndarray, modulus=360):
    """Helper function to convert "coming from" convention to "going to"."""
    going_to = (coming_from + 180) % modulus
    return going_to


def going_to_coming(going_to: np.ndarray, modulus=360):
    """Helper function to convert "going to" convention to "coming from"."""
    coming_from = (going_to - 180) % modulus
    return coming_from


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def angular_frequency_to_frequency(angular_frequency):
    """Helper function to convert angular frequency (omega) to frequency (f)"""
    return angular_frequency / (2 * np.pi)


def _append_last(arr: np.ndarray):
    """Helper function to append the last value of an array to itself."""
    return np.append(arr, arr[-1])

