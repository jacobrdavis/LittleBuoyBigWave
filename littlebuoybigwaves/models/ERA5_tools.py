"""
Author: @jacobrdavis

A collection of functions for working with ECMWF ERA5 reanalysis model data. 
    * https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
    * https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
For information on setting up and using the API, visit:
    * https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS

#TODO:
    - implement a model dict object that can be passed to each function w/o need for input every time
    - figure out why not all vars get loaded (e.g. only wind or only wave can be loaded)
        * https://github.com/ecmwf/cfgrib/issues/111
"""
__all__ = [
    'read_ERA5_gribfile',
    'generate_ERA5_API_request_dict', 
    'generate_ERA5_API_request_filename',
    'generate_ERA5_daterange',
    'ERA5_API_request',
    'get_ERA5_ymdh',
    'get_ERA5_variables',
    'get_ERA5_extent',
    'match_buoy_and_ERA5',
]


from ast import Return
import pandas as pd
import xarray as xr
import numpy as np
from typing import Iterable, List
import cdsapi
import json
from datetime import datetime, timedelta
import cfgrib
import warnings
from littlebuoybigwaves.geo import get_extent

# import datetime
#%%
#%%
def read_ERA5_gribfile(gribfile):
    """
    TODO:_summary_
    https://confluence.ecmwf.int/display/UDOC/How+to+install+ecCodes+with+Python+bindings+in+conda+-+ecCodes+FAQ
    Arguments:
        - gribfile (_type_), _description_

    Returns:
        - (_type_), _description_
    """
    
    # data = list()
    print(f'Loading {gribfile}')
    dataset = xr.open_dataset(gribfile, engine='cfgrib') # NOTE: pynio may be faster engine
    # varKeys = ['time','latitude','longitude'] + list(dataset.keys()) 
    # data.append({key : dataset[key].data for key in varKeys})

    return dataset

def generate_ERA5_API_request_dict(ERA5_vars, year, month, day, time, area):
    
    if day == 'all':
        day = ['01', '02', '03','04', '05', '06', '07', '08', '09', '10', '11', '12',
               '13', '14', '15','16', '17', '18', '19', '20', '21', '22', '23', '24',
               '25', '26', '27','28', '29', '30', '31']
    if month == 'all':
        month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    if time == 'all':
        time = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']

    if area == 'all':
        area = [90, -180, -90, 180]

    if type(day) is not list:
        day = [day]

    if type(month) is not list:
        month = [month]

    if type(year) is not list:
        year = [year]

    requestDict = {
    'product_type': 'reanalysis',
    'variable': ERA5_vars,
    'year' : year,
    'month': month,
    'day'  : day,
    'time': time,
    'area': area,
    'format': 'grib',
    }
    
    # jsonRequest = json.dumps(d)

    return requestDict

def generate_ERA5_API_request_filename(requestDict, prefix = '', postfix = ''):
    
    if len(requestDict['day']) == 1:
        dateRangeStr = requestDict['year'][0] + requestDict['month'][0] + requestDict['day'][0] 
    
    elif len(requestDict['day']) > 1 and len(requestDict['day']) < 31:
        dateRangeStr = requestDict['year'][0] + requestDict['month'][0] + requestDict['day'][0] + \
                       '_to_' + \
                       requestDict['year'][0] + requestDict['month'][0] + requestDict['day'][-1]
    else:
        if len(requestDict['month']) == 1 and len(requestDict['year']) == 1:
            dateRangeStr = requestDict['year'][0] + requestDict['month'][0]

        elif len(requestDict['month']) > 1 and len(requestDict['year']) == 1:
            dateRangeStr = requestDict['year'][0] + requestDict['month'][0] + '_to_' + \
                        requestDict['year'][0] + requestDict['month'][-1]
                        
        elif len(requestDict['year']) > 1:
            dateRangeStr = requestDict['year'][0] + '_to_' + requestDict['year'][-1]

    if requestDict['area'] == [90, -180, -90, 180]:
        area = 'global'
    elif requestDict['area'] == [0, -180, -90, 180]:
        area = 'gsouth'
    elif requestDict['area'] == [90, -180, 0, 180]:
        area = 'gnorth'
    else:
        area = str(requestDict['area'])
        # '(' + ','.join([str(coord) for coord in requestDict['area']]) + ')' 

    productType = requestDict['product_type']

    filename = f'{prefix}ERA5_{productType}_{area}_{dateRangeStr}{postfix}'
    
    return filename

def ERA5_API_request(filename, requestDict):
    """
    
    
    """
    c = cdsapi.Client()
    cdsapi.__doc__
    c.retrieve('reanalysis-era5-single-levels',
                requestDict,
               f'{filename}.grib')
    return

def get_ERA5_variables(keys = 'all'):
    ERA5_vars = {
    'wind_vars' :   ['100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
                    '10m_u_component_of_wind', '10m_v_component_of_neutral_wind', '10m_v_component_of_wind',
                    '10m_wind_gust_since_previous_post_processing', 'instantaneous_10m_wind_gust'],
    'wave_vars' :   ['air_density_over_the_oceans', 'coefficient_of_drag_with_waves', 'free_convective_velocity_over_the_oceans',
                     'maximum_individual_wave_height', 'mean_direction_of_total_swell', 'mean_direction_of_wind_waves',
                     'mean_period_of_total_swell', 'mean_period_of_wind_waves', 'mean_square_slope_of_waves',
                     'mean_wave_direction', 'mean_wave_direction_of_first_swell_partition', 'mean_wave_direction_of_second_swell_partition',
                     'mean_wave_direction_of_third_swell_partition', 'mean_wave_period', 'mean_wave_period_based_on_first_moment',
                     'mean_wave_period_based_on_first_moment_for_swell', 'mean_wave_period_based_on_first_moment_for_wind_waves', 'mean_wave_period_based_on_second_moment_for_swell',
                     'mean_wave_period_based_on_second_moment_for_wind_waves', 'mean_wave_period_of_first_swell_partition', 'mean_wave_period_of_second_swell_partition',
                     'mean_wave_period_of_third_swell_partition', 'mean_zero_crossing_wave_period', 'model_bathymetry',
                     'normalized_energy_flux_into_ocean', 'normalized_energy_flux_into_waves', 'normalized_stress_into_ocean',
                     'ocean_surface_stress_equivalent_10m_neutral_wind_direction', 'ocean_surface_stress_equivalent_10m_neutral_wind_speed', 'peak_wave_period',
                     'period_corresponding_to_maximum_individual_wave_height', 'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_total_swell',
                     'significant_height_of_wind_waves', 'significant_wave_height_of_first_swell_partition', 'significant_wave_height_of_second_swell_partition',
                     'significant_wave_height_of_third_swell_partition', 'wave_spectral_directional_width', 'wave_spectral_directional_width_for_swell',
                     'wave_spectral_directional_width_for_wind_waves', 'wave_spectral_kurtosis', 'wave_spectral_peakedness',
                     'wave_spectral_skewness'],
    'other_vars' :  ['boundary_layer_dissipation', 'charnock', 'eastward_gravity_wave_surface_stress',
                     'eastward_turbulent_surface_stress', 'forecast_surface_roughness', 'friction_velocity',
                     'gravity_wave_dissipation', 'instantaneous_eastward_turbulent_surface_stress', 'instantaneous_northward_turbulent_surface_stress',
                     'land_sea_mask', 'northward_gravity_wave_surface_stress', 'northward_turbulent_surface_stress',
                     'sea_ice_cover', 'u_component_stokes_drift', 'v_component_stokes_drift',],
    'temp_and_pressure_vars' : ['2m_dewpoint_temperature', '2m_temperature', 'ice_temperature_layer_1',
                    'ice_temperature_layer_2', 'ice_temperature_layer_3', 'ice_temperature_layer_4',
                    'maximum_2m_temperature_since_previous_post_processing', 'mean_sea_level_pressure', 'minimum_2m_temperature_since_previous_post_processing',
                    'sea_surface_temperature', 'skin_temperature', 'surface_pressure',],
    'hurricane_waves' : ['100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
                    '10m_u_component_of_wind', '10m_v_component_of_neutral_wind', '10m_v_component_of_wind',
                    '10m_wind_gust_since_previous_post_processing', 'air_density_over_the_oceans', 'charnock',
                    'coefficient_of_drag_with_waves', 'eastward_gravity_wave_surface_stress', 'eastward_turbulent_surface_stress',
                    'forecast_surface_roughness', 'friction_velocity', 'gravity_wave_dissipation',
                    'instantaneous_10m_wind_gust', 'mean_direction_of_total_swell', 'mean_period_of_total_swell',
                    'mean_square_slope_of_waves', 'mean_wave_direction', 'mean_wave_period',
                    'mean_zero_crossing_wave_period', 'model_bathymetry', 'normalized_energy_flux_into_ocean',
                    'normalized_energy_flux_into_waves', 'normalized_stress_into_ocean', 'northward_gravity_wave_surface_stress',
                    'northward_turbulent_surface_stress', 'ocean_surface_stress_equivalent_10m_neutral_wind_direction', 'ocean_surface_stress_equivalent_10m_neutral_wind_speed',
                    'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell',],
    'mss_study' : [ '10m_u_component_of_wind', '10m_v_component_of_wind', 'air_density_over_the_oceans',
                    'charnock', 'coefficient_of_drag_with_waves', 'friction_velocity',
                    'mean_square_slope_of_waves', 'mean_wave_direction', 'mean_wave_period',
                    'significant_height_of_combined_wind_waves_and_swell'],
    'wave_vars_reduced' : ['air_density_over_the_oceans', 'coefficient_of_drag_with_waves', 'mean_square_slope_of_waves',
                    'mean_wave_direction', 'mean_wave_period', 'mean_zero_crossing_wave_period',
                    'model_bathymetry', 'normalized_stress_into_ocean', 'ocean_surface_stress_equivalent_10m_neutral_wind_direction',
                    'ocean_surface_stress_equivalent_10m_neutral_wind_speed', 'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
                    'wave_spectral_directional_width']
    }
    
    if keys != 'all':
        if type(keys) != list:
            keys = [keys]
        ERA5_vars_combined = []
        for key in keys:
            ERA5_vars_combined = ERA5_vars_combined + ERA5_vars[key]
        return ERA5_vars_combined

    else:
        return ERA5_vars


def get_ERA5_extent(lats: Iterable , lons: Iterable) -> List:
    """
    get and round geographic extent

    """
    upperLat = np.max(lats)
    lowerLat = np.min(lats)
    upperLon = np.max(lons)
    lowerLon = np.min(lons)
    
    # ERA5 area
    area = np.array([upperLat, lowerLon, lowerLat, upperLon])
    area = np.round(area*4)/4

    return list(area)

def closest_value(x: float,X: Iterable): # lat,lon,latList, lonList):
    """
    Helper function to find index of closest value of x in list X
    
    """
    Xidx = np.argmin(abs(X-x)) 
    return Xidx    

def hour_rounder(t):
    """
    Anton vBR @ stackoverflow

    https://stackoverflow.com/questions/48937900/round-time-to-nearest-hour-python
    """
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30

    t_rounded = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) +timedelta(hours=t.minute//30))

    return t_rounded

def generate_ERA5_daterange(startDate, endDate, interval = 'days'):
    """
    generate daterange in daily or hourly increments
    
    """
    startDate = hour_rounder(startDate)
    endDate = hour_rounder(endDate)

    if interval == 'days':
        numDays = (endDate - startDate).days
        daterange = [startDate.replace(second=0, microsecond=0, minute=0, hour=0) + \
                    timedelta(days=x) for x in range(numDays+1)]
    
    elif interval == 'hours':
        numHours = int((endDate - startDate).total_seconds() // 3600)
        daterange = [startDate.replace(second=0, microsecond=0, minute=0, hour=0) + \
                    timedelta(hours=x) for x in range(numHours+1)]
    
    return daterange

def sort_str_numbers(strNums):
    """
    Helper function to sort string representation of numbers
    
    """
    if any(':' in s for s in strNums):
        strNumsUnsorted = [s.split(':')[0] for s in strNums] 
    else:
        strNumsUnsorted = strNums.copy()

    sort_index = np.argsort([int(s) for s in strNumsUnsorted])
    strNumsSorted = np.array(strNums)[sort_index]

    return list(strNumsSorted)


def get_ERA5_ymdh(daterange):
    """
    Get years, months, days, and hours from a daterange in string format accepted by ERA5 API.
    
    
    """
    hours = []
    days = []
    months = []
    years = []
    # ymdh = {'years' : [],
    #         'months' : [], 
    #         'days' : [],
    #         'hours' : []}

    for date in daterange:

        hours.append(date.strftime("%H:%M"))
        days.append(date.strftime("%d"))
        months.append(date.strftime("%m"))
        years.append(date.strftime("%Y"))

        # ymdh['hours'].append(date.strftime("%H:%M"))
        # ymdh['days'].append(date.strftime("%d"))
        # ymdh['months'].append(date.strftime("%m"))
        # ymdh['years'].append(date.strftime("%Y"))

    # ymdh_sorted = {key : sort_str_numbers(list(set(ymdh[key])))  for key in ymdh}
    hours = sort_str_numbers(list(set(hours))) # seems to be sorted, but if neccesary ':' can be stripped
    days = sort_str_numbers(list(set(days)))
    months = sort_str_numbers(list(set(months)))
    years = sort_str_numbers(list(set(years)))

    return years, months, days, hours


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



#%%
# ERA5_vars=get_ERA5_variables()

# requestDict = generate_ERA5_API_request_dict(ERA5_vars['wind_vars'],
#                                              year='2021', 
#                                              month=['04'], 
#                                              day=['01'], 
#                                              time='00:00', 
#                                              area= [0,80,0.5,80.5])

# requestFilename = generate_ERA5_API_request_filename(requestDict)

# ERA5_API_request(requestFilename, requestDict)

# fileDir = 'forecast_model_tools/'
# d = read_ERA5_gribfile(requestFilename +'.grib')
#%%
#%%
# def main():
# gribfile ='/Users/jacob/Dropbox/Projects/NHCI/data/ERA5/adaptor.mars.internal-1662834817.4767349-22908-13-e3a87caa-35e4-49f9-8e11-2f01019bd291.grib'
# data = read_ERA5_gribfile(gribfile)

# # data[0]['u10'][0]
# ERA5_vars = ERA5_variables()

# # if __name__ == '__main__':
# #     main()
# js = json.dumps(ERA5_vars)


#%%


# def monthlist(startDate,endDate):
#     # start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
#     total_months = lambda dt: dt.month + 12 * dt.year
#     mlist = []
#     for tot_m in range(total_months(startDate)-1, total_months(endDate)):
#         y, m = divmod(tot_m, 12)
#         # mlist.append(datetime(y, m+1, 1).strftime("%Y-%m"))
#         mlist.append((datetime(y, m+1, 1).year, datetime(y, m+1, 1).month))

#     return mlist

# def sort_str_numbers(strNums):
#     if any(':' in s for s in strNums):
#         strNumsUnsorted = [s.split(':')[0] for s in strNums] 
#     else:
#         strNumsUnsorted = strNums.copy()

#     sort_index = np.argsort([int(s) for s in strNumsUnsorted])
#     strNumsSorted = np.array(strNums)[sort_index]

#     return list(strNumsSorted)

# startDate = datetime(2021,4,1,0,0,0)
# endDate = datetime(2021,8,1,0,0,0)

# monthList = monthlist(startDate,endDate)

#%%

# for y,m in monthList:
#     if y == startDate.year and m == startDate.month:
#         print(y,m)
#         start = startDate
#         end = datetime(y,m+1,1,0,0,0)

#     numHrs = int((end - start).total_seconds()  // 3600)
#     date_list = [start + timedelta(hours=x) for x in range(numHrs+1)]
# #TODO: just do it by day....

# startDate = datetime(2021,4,1,0,0,0)
# endDate = datetime(2021,8,1,0,0,0)
# numDays = (endDate - startDate).days
# dateList = [startDate + timedelta(days=x) for x in range(numDays+1)]

# for date in dateList:
#     # hours = date.strftime("%H:%M")
#     days = date.strftime("%d")
#     months = date.strftime("%m")
#     years = date.strftime("%Y")
#     generate_ERA5_API_request_json(ERA5_vars, years, months, days, times, area)






#%%

# if startDate.year != endDate.year:
#     print('rollover')
#     # TODO: divide into years and repeat this function call for ea one
# else:
#     if startDate.month == endDate.month:
#         months = startDate.strftime("%m")
#         days = monthrange(startDate.year, startDate.month) 
#     elif startDate.month - endDate.month == 1:
#         months = [str(mo) for mo in list(range(startDate.month,endDate.month+1,1))]
#         [startDate.strftime("%m"),endDate.strftime("%m")]
#         _,lastDay = monthrange(startDate.year, startDate.month)  
#     elif startDate.month != endDate.month:
#         months = [str(mo) for mo in list(range(startDate.month,endDate.month+1,1))]
#         # days = 

# startDate.month