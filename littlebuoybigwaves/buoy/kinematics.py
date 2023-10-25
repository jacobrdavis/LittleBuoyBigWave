"""TODO module docstr"""

__all__ = [
    "drift_speed_and_direction",
    "doppler_shift",
]

from typing import Tuple

import pandas as pd
import numpy as np

from littlebuoybigwaves.geo import haversine_distance

KMPH_TO_MPS = 0.277778  # m/s / km/hr


def drift_speed_and_direction(
    longitude: np.array,
    latitude: np.array,
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

    # Calculate drift magnitude and components; convert from km/hr to m/s.
    drift_speed_kmph = dist_km/time_difference_hr
    east_drift_speed_kmph = drift_speed_kmph * np.sin(np.deg2rad(drift_dir_deg))
    north_drift_speed_kmph = drift_speed_kmph * np.cos(np.deg2rad(drift_dir_deg))

    drift_speed_mps = drift_speed_kmph * KMPH_TO_MPS
    east_drift_speed_mps = east_drift_speed_kmph * KMPH_TO_MPS
    north_drift_speed_mps = north_drift_speed_kmph * KMPH_TO_MPS

    # If `append` is truthy, append the last value to maintain input size (n,).
    if append:
        drift_speed_mps = _append_last(drift_speed_mps)
        east_drift_speed_mps = _append_last(east_drift_speed_mps)
        north_drift_speed_mps = _append_last(north_drift_speed_mps)
        drift_dir_deg = _append_last(drift_dir_deg)

    return drift_speed_mps, east_drift_speed_mps, north_drift_speed_mps, drift_dir_deg


def doppler_shift(
    drift_direction_going: np.array,
    wave_direction_coming: np.ndarray,
    drift_speed: np.array,
    intrinsic_frequency: np.ndarray,
    wavenumber: np.ndarray,
) -> Tuple:
    #TODO:

    wave_direction_going = coming_to_going(wave_direction_coming, modulus=360)

    # Compute drift-wave misalignment.
    misalignment_full_deg = drift_direction_going[:, None] - wave_direction_going
    misalignment_deg = (misalignment_full_deg + 180) % 360 - 180

    # Compute the dot product of the drift velocity and wavenumber.
    misalignment_rad = np.deg2rad(misalignment_deg)
    u_dot_k = drift_speed[:, None] * wavenumber * np.cos(misalignment_rad)

    # Adjust the intrinsic frequency by u dot k; note units of rad/s.
    intrinsic_angular_frequency = frequency_to_angular_frequency(intrinsic_frequency)
    absolute_angular_frequency = intrinsic_angular_frequency + u_dot_k
    absolute_frequency = angular_frequency_to_frequency(absolute_angular_frequency)

    return absolute_frequency, u_dot_k, misalignment_deg


def coming_to_going(coming_from: np.array, modulus=360):
    """Helper function to convert "coming from" convention to "going to"."""
    going_to = (coming_from + 180) % modulus
    return going_to


def going_to_coming(going_to: np.array, modulus=360):
    """Helper function to convert "going to" convention to "coming from"."""
    coming_from = (going_to - 180) % modulus
    return coming_from


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def angular_frequency_to_frequency(angular_frequency):
    """Helper function to convert angular frequency (omega) to frequency (f)"""
    return angular_frequency / (2 * np.pi)


def _append_last(arr: np.array):
    """Helper function to append the last value of an array to itself."""
    return np.append(arr, arr[-1])

