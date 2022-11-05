#TODO: put this inside of ERA5/GFS tools!
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable
import warnings

def closest_value(x: float,X: Iterable): # lat,lon,latList, lonList):
    """
    Helper function to find index of closest value of x in list X
    
    """
    Xidx = np.argmin(abs(X-x)) 
    return Xidx    
def get_extent(latitudes: np.array, longitudes: np.array, offset: float = 0):
    """
    #TODO:
    """
    latitudes = np.sort(latitudes)
    longitudes = np.sort(longitudes)

    newExtent = np.array([
        latitudes[0] + offset,
        latitudes[-1] - offset,
        longitudes[0] + offset,
        longitudes[-1] - offset,
    ])
    return newExtent
# METHOD 2: loop through spotters # 
# for spot in spotters:
#     for GFS_run in GFS_dict
#         match spotter and GFSrun
def match_buoy_and_GFSwave(buoy : pd.DataFrame, GFSwave : dict, concatOutput : bool ,prefix : str = ''):
    """
    match GFSwave and buoy locations

    Input:
        - buoy, Pandas dataframe of buoy data 
            * Note the dataframe must have a datetime index and the location must be stored with 
            'latitude' and 'longitude' as the column names
        - GFSwave, a collated list of #TODO:

    """
    matches = {key:[] for key in GFSwave[0].keys()}
    buoyIndices = []
    nLon = np.size(GFSwave[0]['longitude'])
    nLat = np.size(GFSwave[0]['latitude'])

    for GFSrun in GFSwave:
        # timestamp of GFSrun:
        GFStime = GFSrun['time'] 
    #TODO: check for if GFSrun['time'][0] is greater than buoytime[-1]...
        # find index of closest match with the buoy in time:
        buoyIdx = closest_value(GFStime,buoy.index.values)
        buoyTime = buoy.index[buoyIdx].to_numpy()
        
        # time difference:
        timeDiff = buoyTime - GFStime
        
        # check if the time difference is less than one hour:
        if np.abs(timeDiff) < np.timedelta64(1, 'h'):

            # find closest lat and lon:
            GFSlatIdx  = closest_value(buoy.iloc[buoyIdx]['latitude'],GFSrun['latitude'])
            GFSlonIdx  = closest_value(buoy.iloc[buoyIdx]['longitude']+360,GFSrun['longitude'])
            
            # append time and location to dict of matches:
            matches['time'].append(GFSrun['time'])
            matches['latitude'].append(GFSrun['latitude'][GFSlatIdx])
            matches['longitude'].append(GFSrun['longitude'][GFSlonIdx])
            
            # record omitted keys, appending prefix:
            omittedKeys = ['time','latitude','longitude']

            # append all other keys stored in the GFSrun:
            for key in GFSrun.keys() - omittedKeys:
                size = np.size(GFSrun[key])
                if size == nLat*nLon:
                    matches[key].append(GFSrun[key][GFSlatIdx,GFSlonIdx])
                elif size == 3*nLat*nLon:
                    matches[key].append([GFSrun[key][level][GFSlatIdx,GFSlonIdx] for level in range(0,3)])
                else:
                    matches[key].append(GFSrun[key])

            # append the matching index in the buoy dataframe:    
            buoyIndices.append(buoyIdx)

        else: # no match in time
            # print('out of time range')
            continue

    # convert the GFSwave matches dictionary to a single dataframe:
    GFSwave_matches = pd.DataFrame(matches).add_prefix(prefix)

    # extract matching buoy rows from the larger buoy dataframe:
    buoy_matches = buoy.iloc[buoyIndices]

    # TODO: check_timedeltas(datetimeArr1 = buoy_matches.index.tz_convert(None).to_numpy(),
    #                  datetimeArr2 = GFSwave_matches[prefix+'time'].to_numpy())

    if concatOutput == True:
        df_concat = pd.concat([buoy_matches, GFSwave_matches.set_index(buoy_matches.index)], axis=1)
        return df_concat

    else:
        GFSwave_matches.set_index(prefix+'time',inplace=True)
        return buoy_matches, GFSwave_matches

def match_buoy_and_ERA5(
    buoy : pd.DataFrame,
    ERA5 : xr.Dataset, 
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
    spatial_tolerance: float = 0.25,
    bounds: np.ndarray = None,
    concatOutput : bool = True, 
    prefix : str = '',
)-> pd.DataFrame:
    """
    match ERA5 and buoy locations

    Input:
        - buoy, Pandas dataframe of buoy data 
            * Note the dataframe must have a datetime index and the location must be stored with 
            'latitude' and 'longitude' as the column names
        - ERA5, an xarray dataset

    """
    # get ERA5 time, lat, and lon arrays
    t_ERA5   = pd.to_datetime(ERA5['time'].values).values
    lat_ERA5 = ERA5['latitude'].values
    lon_ERA5 = ERA5['longitude'].values

    if bounds is None:
        bounds = get_extent(lat_ERA5, lon_ERA5)

    # get ERA5 keys
    ERA5keys = ['time','latitude','longitude'] + list(ERA5.keys()  - ['meanSea', 'heightAboveGround']) 
    ERA5vars = {key: [] for key in ERA5keys}

    # get buoy time, lat, and lon arrays
    t_buoy = buoy.index.values
    lat_buoy = buoy['latitude'].to_numpy()
    lon_buoy = buoy['longitude'].to_numpy()

    # initilalize lists for matched indices:
    buoy_idx = []
    t_idx = []
    lon_idx = []
    lat_idx = []

    # loop through each buoy dimension and find the closest match in the corresponding ERA5 arrays:
    for t_i, lat_i, lon_i in zip(t_buoy, lat_buoy, lon_buoy):
        # find closest time:
        t_check_idx = closest_value(t_i,t_ERA5)
        lat_check_idx = closest_value(lat_i,lat_ERA5)
        lon_check_idx = closest_value(lon_i,lon_ERA5)

        # if the time diff is w/in 1 hour, append the time, lat, and lon indices as matches:
        time_diff = t_ERA5[t_check_idx] - t_i
        lat_diff = lat_ERA5[lat_check_idx] - lat_i
        lon_diff = lon_ERA5[lon_check_idx] - lon_i
        check1 = np.abs(t_ERA5[t_check_idx] - t_i) <= temporal_tolerance  #TODO: changed from 1 to 0.5 on 10-29-2022
        check2 = np.abs(lat_ERA5[lat_check_idx] - lat_i) <= spatial_tolerance
        check3 = np.abs(lon_ERA5[lon_check_idx] - lon_i) <= spatial_tolerance
        check4 = bounds[0] < lat_i < bounds[1] #TODO: or check lat_i? < extent[1] #TODO: handle edge cases
        check5 = bounds[2] < lon_i < bounds[3]

        if check1 and check2 and check3 and check4 and check5:
            buoy_idx.append(np.where(t_buoy == t_i)[0][0]) #TODO: fix this
            t_idx.append(t_check_idx)
            lat_idx.append(lat_check_idx)
            lon_idx.append(lon_check_idx)
    
    # Extract ERA5 vars via pointwise indexing; see: 
    # https://docs.xarray.dev/en/stable/user-guide/indexing.html#vectorized-indexing
    if len(t_idx) > 0:
        for var in ERA5keys:   
            if var == 'time':
                ERA5vars[var] = ERA5[var].isel(time = xr.DataArray(t_idx, dims="z")).values
            elif var == 'latitude':
                ERA5vars[var] = ERA5[var].isel(latitude = xr.DataArray(lat_idx, dims="z")).values
            elif var == 'longitude':
                ERA5vars[var] = ERA5[var].isel(longitude = xr.DataArray(lon_idx, dims="z")).values
            else:
                ERA5vars[var] = ERA5[var].isel(time = xr.DataArray(t_idx, dims="z"),
                                               latitude = xr.DataArray(lat_idx, dims="z"),
                                               longitude = xr.DataArray(lon_idx, dims="z")).values

    # convert the GFSwave matches dictionary to a single dataframe:
    ERA5_matches = pd.DataFrame(ERA5vars).add_prefix(prefix)
    ERA5_matches.set_index(prefix+'time', inplace=True, drop=False)

    # extract matching buoy rows from the larger buoy dataframe:
    buoy_matches = buoy.iloc[buoy_idx] 

    # check timedeltas
    if len(buoy_matches) > 0:
        check_timedeltas(datetimeArr1 = buoy_matches.index,
                        datetimeArr2 = ERA5_matches.index.tz_localize('UTC'))

    if concatOutput == True:
        df_concat = pd.concat([buoy_matches, ERA5_matches.set_index(buoy_matches.index)], axis=1)
        return df_concat

    else:
        return buoy_matches, ERA5_matches


def check_timedeltas(datetimeArr1, datetimeArr2):
    """
    TODO:

    Arguments:
        - datetimeArr1 (_type_), _description_
        - datetimeArr2 (_type_), _description_
    """

    dt = np.abs(datetimeArr1 - datetimeArr2)
    # mean_dt = dt.mean().seconds / 3600
    # std_dt = dt.std().seconds / 3600
    max_dt = max(dt).seconds / 3600
    # print(f'mean (hr): {round(mean_dt, 2)}; std (hr): {round(std_dt, 2)}; max (hr): {round(max_dt, 2)}')

    if max_dt > 0.5:
        warnings.warn("Warning: maximum dt is larger than 30 minutes.")
