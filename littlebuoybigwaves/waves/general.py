"""
General water wave functions.
"""

# TODO:
# - unit tests
# - rename/unify functions and variables

__all__ = [
    'dispersion',
    'deep_water_dispersion',
    'shallow_water_dispersion',
    'group_to_phase_ratio',
    'depth_regime',
    'intrinsic_group_velocity',
    'intrinsic_dispersion',
    'phase_velocity',
]

from typing import Tuple

import numpy as np
from scipy.optimize import newton

GRAVITY = 9.81  # m/s^2

#TODO: rename to inverse_dispersion.  See pywsra
def dispersion(
    frequency: np.ndarray,
    depth: np.ndarray,
    use_limits: bool = False
) -> np.ndarray:
    """Solve the linear dispersion relationship for the wavenumber, k.

    Given frequencies (in Hz) and water depths, solve the linear dispersion
    relationship for the corresponding wavenumbers, k. Uses a Newton-Rhapson
    root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths, or a `frequency` of shape
        (f,) and `depth` of shape (d,). If the latter, the inputs will be
        meshed to (d,f) ndarrays, assuming a uniform frequency vector for every
        depth provided. The input `frequency` is the frequency in Hz and NOT
        the angular frequency, omega or w.

        `use_limits` might provide speed-up for very large f*d.

    Args:
        frequency (np.ndarray): of shape (f,) or (d,f) containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.
        depth (np.ndarray): of shape (d,) or (d,f) containing water depths.
        use_limits (bool, optional): solve the dispersion relation only where
            kh is outside of the deep and shallow water limits.

    Raises:
        ValueError: if `frequency` and `depth` are not of size (d,f) or of size
            (f,) and (d,), respectively.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """

    #TODO: use doppler-shifted

    frequency = np.asarray(frequency)
    depth = np.asarray(depth)

    # Check incoming shape; if 1-dimensional, map to an (f, d) mesh. Otherwise
    # the shape should already be (f, d). Raise exception for mixed shapes.
    if frequency.ndim == 1 and depth.ndim == 1:
        f = len(frequency)
        d = len(depth)
        frequency = np.tile(frequency, (d, 1))
        depth = np.tile(depth, (f, 1)).T

    elif frequency.ndim == 2 and depth.ndim == 1:
        d, f = frequency.shape
        depth = np.tile(depth, (f, 1)).T

    elif frequency.ndim == 1 and depth.ndim == 2:
        d, f = depth.shape
        frequency = np.tile(frequency, (d, 1))

    elif frequency.shape == depth.shape:
        pass

    else:
        # if frequency.shape != depth.shape:
        raise ValueError(
            '`frequency` and `depth` must be either arrays of size '
            '(f,) and (d,) \n or ndarrays of the same shape. Given:'
            f' frequency.shape={frequency.shape}'
            f' and depth.shape={depth.shape}.')

    if use_limits:
        wavenumber = _dispersion_with_limits(frequency, depth)
    else:
        wavenumber = _dispersion_solver(frequency, depth)

    return wavenumber


def _dispersion_with_limits(frequency, depth):
    """ Solve the dispersion relation only where parameters are outside of the
    deep and shallow water limits.

    Approximates the wavenumber using both the deep and shallow water linear
    dispersion limits and checks against the `kh` limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    Frequencies and depths outside of these limits are solved using
    a standard root-finding algorithm. This might provide speed-up for cases
    where the combined size of the number of depths and frequencies is very
    large, e.g., O(10^6) and above, since an iterative approach is not needed
    for `kh` at the tanh(kh) limits. Values close to the limits will be
    approximate.

    Args:
        frequency (np.ndarray): of shape (d,f) containing frequencies in [Hz].
        depth (np.ndarray): of shape (d,f) containing water depths.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """
    wavenumber = np.empty(frequency.shape)

    wavenumber_shallow = shallow_water_dispersion(frequency, depth)
    wavenumber_deep = deep_water_dispersion(frequency)

    in_deep = wavenumber_deep * depth > np.pi
    in_shallow = wavenumber_shallow * depth < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)

    wavenumber_intermd = _dispersion_solver(frequency[in_intermd],
                                            depth[in_intermd])

    wavenumber[in_deep] = wavenumber_deep[in_deep]
    wavenumber[in_shallow] = wavenumber_shallow[in_shallow]
    wavenumber[in_intermd] = wavenumber_intermd

    return wavenumber


def deep_water_dispersion(frequency):
    """Computes wavenumber from the deep water linear dispersion relationship.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    deep water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the deep water limit, tanh(kh) -> 1, has the
    closed form solution k = omega^2 / g and is (approximately) valid for
    kh > np.pi (h > L/2).

    Args:
        frequency (np.ndarray): of any shape containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.

    Returns:
        np.ndarray: (of shape equal to the input shape) containing wavenumbers.
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency**2 / GRAVITY


def shallow_water_dispersion(frequency, depth):
    """Computes wavenumber from shallow water linear dispersion.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    shallow water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the shallow water limit, kh -> kh, has the
    closed form solution k = omega / sqrt(gh) and is (approximately) valid for
    kh < np.pi/10 (h < L/20).

    Args:
        frequency (np.ndarray): of shape (d,f) containing frequencies in [Hz].
            NOT the angular frequency, omega or w.
        depth (np.ndarray): of shape (d,f) containing water depths.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency / np.sqrt(GRAVITY * depth)


def _dispersion_solver(frequency: np.ndarray, depth: np.ndarray) -> np.ndarray:
    r"""Solve the linear dispersion relationship.

    Solves the linear dispersion relationship w^2 = gk tanh(kh) using a
    Scipy Newton-Raphson root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths. The input `frequency` is
        the frequency in Hz and NOT the angular frequency, omega or w.

    Args:
        frequency (np.ndarray): of shape (d,f) containing frequencies in [Hz].
        depth (np.ndarray): of shape (d,f) containing water depths.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """

    angular_frequency = frequency_to_angular_frequency(frequency)

    wavenumber_deep = deep_water_dispersion(frequency)

    wavenumber = newton(func=_dispersion_root,
                        x0=wavenumber_deep,
                        args=(angular_frequency, depth),
                        fprime=_dispersion_derivative)
    return wavenumber


def _dispersion_root(wavenumber, angular_frequency, depth):
    #TODO:
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def group_to_phase_ratio(
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Compute the ratio of group velocity to phase velocity.

    Note: to prevent overflows in `np.sinh`, the product of wavenumber and
    depth (relative depth) are used to assign ratios at deep or shallow limits:

        shallow:  Cg = 1.0 if kh < np.pi/10 (h < L/20)
           deep:  Cg = 0.5 if kh > np.pi    (h > L/2)

    Args:
        wavenumber (np.ndarray): of shape (k,) containing wavenumbers
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: of shape (k,) containing ratio at each wavenumber.
    """
    kh = wavenumber * depth
    in_deep, in_shallow, in_intermd = depth_regime(kh)
    ratio = np.empty(kh.shape)
    ratio[in_deep] = 0.5
    ratio[in_shallow] = 1.0
    ratio[in_intermd] = 0.5 + kh[in_intermd] / np.sinh(2 * kh[in_intermd])
    return ratio


def depth_regime(kh: np.ndarray) -> Tuple:
    """ Classify depth regime based on relative depth.

    Classify depth regime based on relative depth (product of wavenumber
    and depth) using the shallow and deep limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    The depth regime is classified as intermediate if not at the deep or
    shallow limits.

    Args:
        kh (np.ndarray): relative depth of shape (k, )

    Returns:
        np.ndarray[bool]: true where kh is deep, false otherwise
        np.ndarray[bool]: true where kh is shallow, false otherwise
        np.ndarray[bool]: true where kh is intermediate, false otherwise
    """
    in_deep = kh > np.pi
    in_shallow = kh < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)
    return in_deep, in_shallow, in_intermd


def intrinsic_group_velocity(wavenumber, frequency=None, depth=np.inf):
    ratio = group_to_phase_ratio(wavenumber, depth)
    return ratio * phase_velocity(wavenumber, frequency, depth)


def intrinsic_dispersion(wavenumber, depth=np.inf):
    GRAVITY = 9.81
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return np.sqrt(gk * np.tanh(kh))  # angular frequency


def phase_velocity(wavenumber, frequency=None, depth=np.inf):
    if frequency is None:
        angular_frequency = intrinsic_dispersion(wavenumber, depth)
    else:
        angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency / wavenumber


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency
