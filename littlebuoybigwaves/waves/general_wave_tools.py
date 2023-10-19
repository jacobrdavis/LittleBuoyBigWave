"""
A collection of general water wave functions.

#TODO:
- Tests of dispersion functions



"""
__all__ = [
    'dispersion',
    'deep_water_dispersion',
    'shallow_water_dispersion',
]

import numpy as np
from scipy.optimize import newton

GRAVITY = 9.81  # m/s^2


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
    return  gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def dispersion_regime(wavenumber, depth):
    #TODO:
    is_deep = kh > np.pi
    is_shallow = kh < np.pi/10


#%%
#%%
# import matplotlib.pyplot as plt
#TODO: testing
# f = np.ones(1000)*0.1
# w = 2*np.pi*f
# h = np.linspace(0.5, 1000, len(f))
# k = dispersion(f, h)

# fig, ax = plt.subplots()
# ax.plot(h, k[:, 0], color='k')
# ax.axhline(w[0]**2 / GRAVITY)
# ax.plot(h, w * np.sqrt(1 / (GRAVITY * h)))

# #%%
# f = np.linspace(0.05, 0.5, 40)

# h = np.linspace(0.5, 10, 10)

# f = 0.05
# h = 0.5
# k = dispersion(f, h)

# k
# #%%
# import time
# n = 10
# f = np.linspace(0.05, 0.5, 40)
# f_mat = np.tile(f, (n, 1))
# # w = 2*np.pi*f
# # w_mat = 2*np.pi*f_mat
# h = np.linspace(0.5, n, len(f_mat))
# h_mat = np.tile(h, (len(f), 1)).T

# start = time.time()
# wavenumber_1 = dispersion(frequency=f, depth=h, use_limits=False)
# end = time.time()
# print(end - start)


# # start = time.time()
# # wavenumber = dispersion(frequency=f_mat, depth=h, use_limits=False)
# # end = time.time()
# # print(end - start)

# start = time.time()
# wavenumber_2 = dispersion(frequency=f_mat, depth=h_mat, use_limits=True)
# end = time.time()
# print(end - start)

# start = time.time()
# wavenumber_3 = dispersion(frequency=f_mat, depth=h_mat, use_limits=False)
# end = time.time()
# print(end - start)

# print(np.mean((wavenumber_2 - wavenumber_3)**2))
# print(wavenumber_1)
# print(wavenumber_2)
# print(wavenumber_3)
# #%%

# fig, ax = plt.subplots()
# ax.plot(h, wavenumber_1[:, 0])
# ax.plot(h, wavenumber_2[:, 0])