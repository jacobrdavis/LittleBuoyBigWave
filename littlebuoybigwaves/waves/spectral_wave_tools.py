"""
A collection of spectral water wave functions.
#TODO: UNITTESTS
"""

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

import numpy as np

from typing import Tuple


def mean_square_slope(
    energy: np.ndarray,
    freq: np.ndarray,
    freq_range='dynamic',
    norm=None,
    direction=None,
    spread=None,
) -> Tuple:
    """
    Compute mean square slope (mss) as 4th moment of spectral tail.

    Args:
        - energy, (np.ndarray) energy densities [n,m] (m^2/Hz).
        - freq, (np.ndarray) frequencies [n,m] (Hz).
        - freq_range, keyword or literal specifying integration range.
            Accepted values:
            * 'dynamic', integrate over energy frequencies f_e to 2*f_e
            * 'total', integrate from min(freq) to max(freq)
            * `list`, `tuple`, or `ndarray` of freq limits, e.g. [0.1,0.5]
        - norm, (str or None) keyword specifying normalization type.
            Accepted values:
            * 'frequency', normalize by bandwidth of `freq` where `freq`
               is in `freq_range` (divide by max-min)
            * 'direction', normalize by 'direction' where where `freq`
               is in `freq_range` (divide element-wise)
            * 'spread', normalize by 'spread' where where `freq` is
               in `freq_range` (divide element-wise)
        - direction, (np.ndarray) spectral directions [n,m] (rad).
            * Used only if `norm` = 'direction'
        - spread, (np.ndarray) spectral directional spread [n,m] (rad).
            * Used only if `norm` = 'spread'

    Returns:
        - mss, (float) mean square slope [1,]
        - bandwidth, (float) difference of mss integral limits [1,]
        - freq_range_logical, (nd.array[bool]) logical array
          corresponding to valid frequencies in specified range [n,1]

    Example:
    
    Compute normalized mss over the static range 0.1 Hz to 0.5 Hz
        mss = mean_square_slope(energy, freq, norm=True, freq_range=[0.1, 0.5])

    """
    VALID_NORMS = ['frequency', 'direction', 'spread', None]
    VALID_FREQ_RANGES = ['dynamic', 'total', list, tuple, np.ndarray]
    ACC_GRAV = 9.81 # (m/s)

    # if inputs are lists, cast them as arrays:
    energy = np.asarray(energy)
    freq = np.asarray(freq)
    
    # Handle ndarrays
    if energy.ndim > 1:
        return _handle_mean_square_slope_ndarray(energy, freq, 
                                                freq_range=freq_range,
                                                norm=norm,
                                                direction=direction,
                                                spread=spread)
    # Convert NoneType norm to an empty list and raise an error if it is
    # an invalid value. If multiple norms are specified, every element
    # must be checked against VALID_NORMS.
    if norm is None:
        norm = []

    if norm:
        if isinstance(norm, list) and len(norm)>1:
            bad_norm = any(n not in VALID_NORMS for n in norm)
        else:
            bad_norm = norm not in VALID_NORMS
        if bad_norm:
            raise ValueError((f'The specified normalization, {norm}, is invalid'
            f' or not supported. The valid options are: {VALID_NORMS}'))


    # Determine logical array corresponding to valid frequencies.
    if freq_range == 'dynamic':
        f_e = energy_period(energy, freq, returnAsFreq = True)
        freq_range_logical = np.logical_and(freq>=f_e, freq<=2*f_e)
    elif freq_range == 'total':
        freq_range_logical = np.logical_and(freq>=np.min(freq), freq<=np.max(freq))
    elif isinstance(freq_range, (list, tuple, np.ndarray)):
        freq_range_logical = np.logical_and(freq>=freq_range[0], freq<=freq_range[1])
    else:
        raise ValueError((f'The specified freq range, {freq_range}, is invalid'
        f' or not supported. The valid options are: {VALID_FREQ_RANGES}'))

    # Compute the fourth moment, bandwidth, and mean square slope.
    # Handle the special cases where directional or spread normalization
    # are specified; in these cases, the energy must be normalized by
    # the frequency-dependent direction prior to integration.
    if 'direction' in norm:
        if direction is not None:
            energy = energy/direction
        else:
            raise ValueError(('Directional normalization specified but no'
            ' direction array provided. Please provide input for `direction`.'))
    if 'spread' in norm:
        if spread is not None:
            energy = energy/spread
        else:
            raise ValueError(('Directional spread normalization specified but no'
            ' spread array provided. Please provide input for `spread`.'))

    fourth_moment =spectral_moment(energy[freq_range_logical],
                                   frequency=freq[freq_range_logical],
                                   n=4)
    mss = ((2*np.pi)**4*fourth_moment)/(ACC_GRAV**2) # (-)
    bandwidth = np.ptp(freq[freq_range_logical])
  
    # Handle the remaining normalization cases.
    if 'frequency' in norm:
        mss = mss/bandwidth
    else:
        pass

    return mss, bandwidth, freq_range_logical


def _handle_mean_square_slope_ndarray(energy, freq, direction=None, spread=None, **kwargs):
    """
    Helper function to handle multidimensional input to the
    `mean_square_slope()` function. Not yet vectorized so it can handle
    ragged arrays...

    Args:
        - energy, (np.ndarray) energy densities [n,m] (m^2/Hz).
        - freq, (np.ndarray) frequencies [n,m] (Hz).

        All remaining arguments are passed to `mean_square_slope()`

    Returns:
        Tuple: of three nd.arrays containing the mss, bandwidth, and
        a boolean array corresponding  to the mss frequency range.
    """
    mss = []
    bandwidth = []
    freq_range_logical = []

    if spread is None and direction is None:
        for freq_i, energy_i in zip(freq, energy):
            out = mean_square_slope(energy_i,
                                    freq_i,
                                    **kwargs)
            mss.append(out[0])
            bandwidth.append(out[1])
            freq_range_logical.append(out[2])

    if spread is None and direction is not None:
        for freq_i, energy_i, direction_i in zip(freq, energy, direction):
            out = mean_square_slope(energy_i,
                                    freq_i,
                                    direction=direction_i,
                                    **kwargs)
            mss.append(out[0])
            bandwidth.append(out[1])
            freq_range_logical.append(out[2])

    if  spread is not None and direction is None:
        for freq_i, energy_i, spread_i in zip(freq, energy, spread):
            out = mean_square_slope(energy_i,
                                    freq_i,
                                    spread=spread_i,
                                    **kwargs)
            mss.append(out[0])
            bandwidth.append(out[1])
            freq_range_logical.append(out[2])
    if  spread is not None and direction is not None:
        for freq_i, energy_i, spread_i, direction_i \
                    in zip(freq, energy, spread, direction):
            out = mean_square_slope(energy_i,
                                    freq_i,
                                    spread=spread_i,
                                    direction=direction_i,
                                    **kwargs)
            mss.append(out[0])
            bandwidth.append(out[1])
            freq_range_logical.append(out[2])

    return np.array(mss), np.array(bandwidth), np.array(freq_range_logical)


def energy_period(energy, freq, returnAsFreq = False):
    """
    Function to compute energy period (centroid period)
    
    Input:
        - energy, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - freq, input array of frequencies ([n,1] arr OR [n,m] ndarr)
        - returnAsFreq, optional boolean flag to return energy frequency rather than its reciprocal
    
    Output:
        - Te, energy period = f_e^(-1) ([1,] float)
            * if energy is empty or invalid, Te is assigned a NaN

    Example:
        Te = energy_period(energy, freq)

    """
    if hasattr(energy, '__len__') and (not isinstance(energy, str)):
        firstMoment = spectral_moment(energy, freq, n=1) # = np.trapz(np.multiply(energy,freq),x=freq)
        area = spectral_moment(energy, freq, n=0) # = np.trapz(energy,x=freq)
        f_e = np.divide(firstMoment,area)
        
        if not 0.05 < f_e < 2:
            warnings.warn((f'The energy frequency, `f_e` = {np.round(f_e,3)}'
            ' Hz, is suspicious...check that you have provided `energy` and'
            ' `freq` in the correct order.'))

        if returnAsFreq == True:
            return f_e

        Te = np.reciprocal(f_e)
    else:
        Te = np.NaN
        warnings.warn('`energy` is empty or invalid; output assigned as NaN.')
    return Te


# def spectral_moment(energy, freq=None, n=0):
#     """
#     Function to compute 'nth' spectral moment
    
#     Input:
#         - energy, input array of energy densities ([n,1] arr OR [n,m] ndarr)
#         - freq, input array of frequencies ([n,1] arr OR [n,m] ndarr)
#         - n, moment ([1,] int)
       
#     Output:
#         - mn, nth spectral moment ([1,] float)
#             * if energy is empty or invalid, mn is assigned a NaN

#     Example:
    
#     Compute 4th spectral moment:
#         m4 = spectral_moment(energy, freq, n=4)
#     """
#     if hasattr(energy, '__len__') and (not isinstance(energy, str)):
#         # m_n = np.trapz(np.multiply(energy,freq**n),x=freq)
#         fn = np.power(freq,n)
#         mn = np.trapz(np.multiply(energy,fn),x=freq)
        
#     else:
#         mn = np.NaN
#     return mn


def spectral_moment(energy_density, frequency, n):
    """
    Compute the 'nth' spectral moment.

    Integrates along the last axis.
    """
    frequency_n = frequency ** n
    moment_n = np.trapz(energy_density * frequency_n, x=frequency, axis=-1)
    return moment_n


def moment_weighted_mean(arr, energy_density, frequency, n):
    moment_n = spectral_moment(energy_density=energy_density, frequency=frequency, n=n)
    weighted_moment_n = spectral_moment(energy_density=energy_density * arr, frequency=frequency, n=n)
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

 