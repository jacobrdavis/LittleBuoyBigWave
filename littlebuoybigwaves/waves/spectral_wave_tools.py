"""
A collection of spectral water wave functions.
"""
__all__ = [
    'mean_square_slope',
    'energy_period',
    'spectral_moment',
    'sig_wave_height',
]

import numpy as np

def mean_square_slope(E, f, norm=False, range='dynamic'):
    """
    Function to compute mean square slope as 4th moment of spectral tail
    
    Input:
        - E, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - f, input array of frequencies ([n,1] arr OR [n,m] ndarr)
        - norm, boolean flag for normalization by frequency range (see Iyer et al. 2022 eq. 4)
        - range, indicator or literal input to indicate mss integration frequency range
            Accepted values:
            * 'dynamic', integrate from fe to 2*fe where 'fe' is the energy frequency
            * 'total', integrate over the entire range from min(f) to max(f)
            * list or tuple of frequency limits, e.g. [0.1,0.5]
            * if none of the above are provided, the default static range 0.2 to 0.4 Hz is used
    
    Output:

    if 'norm' is True
        - mssOut, normalized mean square slope value ([1,] float)
        - bandWidth, frequency range used to calculate mssOut (difference of integral limits) ([1,] float)
        - freqRangeLogical, logical array corresponding to valid frequencies in specified range ([nx1] bool arr) 
    
    if 'norm' is False
        - mssOut, mean square slope value ([1,] float)

    Example:
    
    Compute normalized mss over the static range 0.1 Hz to 0.5 Hz
        mss = mean_square_slope(E, f, norm=True, range=[0.1, 0.5])

    """
    # if inputs are lists, cast them as arrays:
    E = np.asarray(E)
    f = np.asarray(f)
    
    #TODO: numpy stack?


    # Handle ndarrays
    #TODO: put into a fun?
    if len(E.shape) >= 2:
        if norm == True: #TODO: make this work
            mssOut = []
            bandWidth = []
            freqRangeLogical = []

            for f_i,E_i in zip(f,E): #TODO: vectorize? maybe too ragged...
                mss_i, bandWidth_i, freqRangeLogical_i = mean_square_slope(E_i, f_i, norm, range)#TODO: dont specify opts
                mssOut.append(mss_i)
                bandWidth.append(bandWidth_i)
                freqRangeLogical.append(freqRangeLogical_i)
                
            return np.array(mssOut), np.array(bandWidth), np.array(freqRangeLogical)

        else:
            mssOut = np.array([mean_square_slope(Ei, fi, norm, range) for fi,Ei in zip(f,E)])
            return mssOut

    # compute logical array corresponding to valid frequencies in specified range (or range type)
    if range == 'dynamic':
        # fe  = np.divide(np.sum(np.multiply(f,E)),np.sum(E))
        fe = energy_period(E, f, returnAsFreq = True)
        freqRangeLogical = np.logical_and(f>=fe, f<=2*fe)

    elif range == 'total':
        freqRangeLogical = np.logical_and(f>=np.min(f), f<=np.max(f))

    elif type(range) is list or type(range) is tuple:
        freqRangeLogical = np.logical_and(f>=range[0], f<=range[1])

    else:
        freqRangeLogical = np.logical_and(f>=0.2,f<=0.4)
    
    fourthMoment =spectral_moment(E[freqRangeLogical], f=f[freqRangeLogical], n=4)  # fourthMoment = np.sum((f[freqRangeLogical])**4 * E[freqRangeLogical] ) * df  # NOTE: spotter appears to have non-constant df 
    g = 9.81 # (m/s)
    mss = ((2*np.pi)**4*fourthMoment)/(g**2) # (-)

    if norm == True:
        bandWidth = np.ptp(f[freqRangeLogical])
        mssOut = mss/bandWidth
        return mssOut, bandWidth, freqRangeLogical

    else:
        mssOut = mss
        return mssOut
    
def energy_period(E, f, returnAsFreq = False):
    """
    Function to compute energy period (centroid period)
    
    Input:
        - E, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - f, input array of frequencies ([n,1] arr OR [n,m] ndarr)
        - returnAsFreq, optional boolean flag to return energy frequency rather than its reciprocal
    
    Output:
        - Te, energy period = fe^(-1) ([1,] float)
            * if E is empty or invalid, Te is assigned a NaN

    Example:
        Te = energy_period(E, f)

    """
    if hasattr(E, '__len__') and (not isinstance(E, str)):
        firstMoment = spectral_moment(E, f, n=1) # = np.trapz(np.multiply(E,f),x=f)
        area = spectral_moment(E, f, n=0) # = np.trapz(E,x=f)
        fe = np.divide(firstMoment,area)
        
        if returnAsFreq == True:
            return fe

        Te = np.reciprocal(fe)
    else:
        Te = np.NaN
    return Te

def spectral_moment(E, f=None, n=0):
    """
    Function to compute 'nth' spectral moment
    
    Input:
        - E, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - f, input array of frequencies ([n,1] arr OR [n,m] ndarr)
        - n, moment ([1,] int)
       
    Output:
        - mn, nth spectral moment ([1,] float)
            * if E is empty or invalid, mn is assigned a NaN

    Example:
    
    Compute 4th spectral moment:
        m4 = spectral_moment(E, f, n=4)
    """
    if hasattr(E, '__len__') and (not isinstance(E, str)):
        # m_n = np.trapz(np.multiply(E,f**n),x=f)
        fn = np.power(f,n)
        mn = np.trapz(np.multiply(E,fn),x=f)
        
    else:
        mn = np.NaN
    return mn

def sig_wave_height(E, f):
    """
    Function to compute significant wave height
    
    Input:
        - E, input array of energy densities ([n,1] arr OR [n,m] ndarr)
        - f, input array of frequencies ([n,1] arr OR [n,m] ndarr)
       
    Output:
        - Hs, significant wave height ([1,] float)
            * if E is empty or invalid, Hs is assigned a NaN

    Example:
    
        Hs = sig_wave_height(E, f)

    """
    if hasattr(E, '__len__') and (not isinstance(E, str)):
        # recover variance from spectrum:
        m0 = spectral_moment(E, f, n=0) 

        # standard deviation:
        stDev = np.sqrt(m0)  

        # sig wave height:   
        Hs  =  4*stDev
      
    else:
        Hs = np.NaN
    return Hs


#%% testing
def main():
    E = [0.1, 0.5, 10, 0.5]
    f = [0.05, 0.1, 0.25, 0.5]
    range = [0.05,0.5]
    range = 'total'
    mss = mean_square_slope(E, f, norm=False, range=range)
    print(mss)

if __name__=='__main__':
    main()
#%%

 