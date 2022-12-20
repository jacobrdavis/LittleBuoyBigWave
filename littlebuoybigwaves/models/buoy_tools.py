
from typing import Iterable

import numpy as np
import scipy


def closest_value(x: float,X: Iterable): # lat,lon,latList, lonList):
    """
    Helper function to find index of closest value of x in list X
    
    """
    Xidx = np.argmin(abs(X-x))
    return Xidx

def match_model_and_buoy_by_interpolation(
    buoy: dict,
    model: dict,
):
    """
    Match model and buoy observations using linear interpolation in time
    and bilinear interpolation in space.

    Args:
        buoy (dict): dictionary containing buoy coordinates 'time',
        'latitude', and 'longitude'; key names much match exactly.

        model (dict): dictionary containing models coordinates 'time',
        'latitude', and 'longitude' plus an additional key-value pair
        containing 'field', the field variable to be matched onto the
        buoy coordinates; key names much match exactly.
    """
    t_sort_indices = np.searchsorted(model('time'), buoy('time'))

    field_matches = []

    points = (model('latitude'), model('longitude'))
    for i, j in enumerate(t_sort_indices):

        x_i = (buoy('latitude')[i], buoy('longitude')[i]) 

        values_jm1 = model('field').values[j-1] # left
        values_j = model('field').values[j] # right

        bilinear_value_jm1 = scipy.interpolate.interpn(points, values_jm1, x_i, method='linear')
        bilinear_value_j = scipy.interpolate.interpn(points, values_j, x_i, method='linear')

        value = np.interp(buoy('time')[i].astype("float"),
                        np.array([model('time')[j-1], model('time')[j]]).astype("float"),
                        np.concatenate([bilinear_value_jm1, bilinear_value_j]))
        field_matches.append(value)

    return np.array(field_matches)


def match_model_and_buoy_by_nearest(
    buoy: dict,
    model: dict,
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
    spatial_tolerance: float = 0.25,
):
    """
    Match model and buoy observations using the nearest model point to
    the current observation.

    Args:
        buoy (dict): dictionary containing buoy coordinates 'time',
        'latitude', and 'longitude'; key names much match exactly.

        model (dict): dictionary containing models coordinates 'time',
        'latitude', and 'longitude' plus an additional key-value pair
        containing 'field', the field variable to be matched onto the
        buoy coordinates; key names much match exactly.

        temporal_tolerance (np.timedelta64, optional): maximum allowable
        time difference between a model and observation point. Defaults
        to np.timedelta64(30, 'm').
       
        spatial_tolerance (float, optional): maximum allowable spatial
        difference between a model and observation point. Defaults to
        0.25 degrees.
    """
    # initilalize lists for matched indices:
    index_matches = {k: [] for k in ['buoy', 'time', 'latitude', 'longitude']}
    matches = {k: [] for k in ['time', 'latitude', 'longitude', 'field']}

    for t_i, lat_i, lon_i in zip(buoy['time'], buoy['latitude'], buoy['longitude']):
        # find closest time:
        t_check_idx = closest_value(t_i,model['time'])
        lat_check_idx = closest_value(lat_i,model['latitude'])
        lon_check_idx = closest_value(lon_i,model['longitude'])

        # if the time diff is w/in 1 hour, append the time, lat, and lon indices as matches:
        check1 = np.abs(model['time'][t_check_idx] - t_i) <= temporal_tolerance
        check2 = np.abs(model['latitude'][lat_check_idx] - lat_i) <= spatial_tolerance
        check3 = np.abs(model['longitude'][lon_check_idx] - lon_i) <= spatial_tolerance
        # check4 = bounds[0] < lat_i < bounds[1] #TODO: or check lat_i? < extent[1] #TODO: handle edge cases
        # check5 = bounds[2] < lon_i < bounds[3]

        if check1 and check2 and check3:
            index_matches['buoy'].append(np.where(buoy['time'] == t_i)[0][0])
            index_matches['time'].append(t_check_idx)
            index_matches['latitude'].append(lat_check_idx)
            index_matches['longitude'].append(lon_check_idx)

    if len(index_matches['time']) > 0:
        matches['time'] = model['time'][index_matches['time']]
        matches['latitude'] = model['latitude'][index_matches['latitude']]
        matches['longitude'] = model['longitude'][index_matches['longitude']]
        matches['field'] = model['field'].values[index_matches['time'],
                                                 index_matches['latitude'],
                                                 index_matches['longitude']]

    return np.array(matches)