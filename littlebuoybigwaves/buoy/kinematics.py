"""
Kinematics functions.
"""

# TODO:
# - rename/unify functions and variables


__all__ = [
    "drift_speed_and_direction",
    "doppler_shift",
    "doppler_correct",
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


def doppler_correct(
    drift_direction_going: np.ndarray,
    wave_direction_coming: np.ndarray,
    drift_speed: np.ndarray,
    intrinsic_frequency: np.ndarray,
    wavenumber: np.ndarray,
) -> Tuple:
    #TODO: specify shapes...
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

