
from typing import Iterable

import numpy as np
import scipy


def closest_value(x: float,X: Iterable): # lat,lon,latList, lonList):
    """ Helper function to find index of closest value of x in list X."""
    Xidx = np.argmin(abs(X-x))
    return Xidx


def _check_array_is_sorted(arr):
    """Helper function that returns True if an array is sorted in either
    ascending or descending order.
    """
    difference = np.diff(arr)
    difference_flip = np.diff(np.flip(arr))
    zero = np.array(0, dtype=difference.dtype)
    is_ascending = np.all(difference >= zero)
    is_descending = np.all(difference_flip >= zero)
    return np.any([is_ascending, is_descending])


def _check_match_model_and_buoy_by_interpolation_inputs(buoy, model):
    """Helper function to check inputs of the match model and buoy function
    by interpolation are sorted.
    """
    buoy_time_is_sorted = _check_array_is_sorted(buoy['time'])

    if buoy_time_is_sorted:
        pass
    else:
        raise ValueError(f"`buoy` time is not sorted.")

    model_time_is_sorted = _check_array_is_sorted(model['time'])

    if model_time_is_sorted:
        pass
    else:
        raise ValueError(f"`model` time is not sorted.")

    # for coordinate in ['time', 'latitude', 'longitude']:
    #     coordinate_is_sorted = _check_array_is_sorted(buoy[coordinate])

    #     if coordinate_is_sorted:
    #         pass
    #     else:
    #         raise ValueError(f"`buoy` coordinate {coordinate} is not sorted.")

    # for coordinate in ['time', 'latitude', 'longitude']:
    #     coordinate_is_sorted = _check_array_is_sorted(model[coordinate])

    #     if coordinate_is_sorted:
    #         pass
    #     else:
    #         raise ValueError(f"`model` coordinate {coordinate} is not sorted.")


def match_model_and_buoy_by_interpolation(
    buoy: dict,
    model: dict,
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
):
    """
    Match model and buoy observations using linear interpolation in time
    and bilinear interpolation in space.

    Note:
        Input dictionaries must contain the following items:

        `buoy` key-value pairs:
        - 'time': np.array[datetime64] with shape (b,)
        - 'latitude': np.array[float] with shape (b,)
        - 'longitude': np.array[float] with shape (b,)

        `model` key-value pairs:
        - 'time': np.array[datetime64] with shape (t,)
        - 'latitude': np.array[float] with shape (m,)
        - 'longitude': np.array[float] with shape (n,)
        - 'field': np.array[float] with shape (t,m,n)

        Where the additional 'field' variable in `model` contains the values
        to be matched onto the buoy coordinates. Time coordinates must be
        sorted and key names much match exactly.

    Args:
        buoy (dict): dictionary containing buoy coordinates 'time',
            'latitude', and 'longitude' (see above).
        model (dict): dictionary containing the model coordinates 'time',
            'latitude', and 'longitude' and values'field' (see above).
        temporal_tolerance (np.timedelta64, optional): max allowable time delta
            between model and buoy times. Defaults to np.timedelta64(30, 'm').

    Returns:
        np.array: of shape (b, ) the field variable interpolated onto the buoy
            time and coordinates.
    """
    _check_match_model_and_buoy_by_interpolation_inputs(buoy, model)

    t_sort_indices = np.searchsorted(model['time'], buoy['time'])

    field_matches = []

    points = (model['latitude'], model['longitude'])
    for i, j in enumerate(t_sort_indices):

        if j < len(model['time']):
            time_difference = np.abs(buoy['time'][i] - model['time'][j])
        else:
            time_difference = None #TODO: do not love this, but try it

        if time_difference is None or time_difference > temporal_tolerance:
            value = np.nan
        else:
            x_i = (buoy['latitude'][i], buoy['longitude'][i])

            field_values_jm1 = model['field'][j-1]  # left
            field_values_j = model['field'][j]  # right

            bilinear_value_jm1 = scipy.interpolate.interpn(points,
                                                           field_values_jm1,
                                                           x_i, 
                                                           method='linear',
                                                           bounds_error=False,
                                                           fill_value=np.NaN)

            bilinear_value_j = scipy.interpolate.interpn(points,
                                                         field_values_j,
                                                         x_i,
                                                         method='linear',
                                                         bounds_error=False,
                                                         fill_value=np.NaN)

            value = np.interp(buoy['time'][i].astype("float"),
                              np.array([model['time'][j-1],
                                        model['time'][j]]).astype("float"),
                              np.concatenate([bilinear_value_jm1,
                                              bilinear_value_j]))

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
        'latitude', and 'longitude' where
            'time': np.array[datetime64]
            'latitude': np.array[float]
            'longitude': np.array[float]
        All arrays should be sorted and key names much match exactly.

        model (dict): dictionary containing models coordinates 'time',
        'latitude', and 'longitude' plus an additional key-value pair
        containing 'field' (np.array), the field variable to be matched
        onto the buoy coordinates; key names much match exactly.

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
        matches['field'] = model['field'][index_matches['time'],
                                          index_matches['latitude'],
                                          index_matches['longitude']]

    return matches

    # for i, j, k, l in enumerate(t_sort_indices, lat_sort_indices, lon_sort_indices):

    # ####
    # #TODO: shows it is 'ij' indexing:
    # # LAT, LON = np.meshgrid(
    # #     np.array(lat_model[lat_sort_indices[i]-1:lat_sort_indices[i]+1]),
    # #     np.array(lon_model[lon_sort_indices[i]-1:lon_sort_indices[i]+1]),
    # #     indexing='ij'
    # # )

    # x_i = (lat_spot[i], lon_spot[i])

    # # (lat,lon))
    # points = (
    #     np.array(lat_model[lat_sort_indices[i]-1:lat_sort_indices[i]+1]),
    #     np.array(lon_model[lon_sort_indices[i]-1:lon_sort_indices[i]+1]),
    # )

    # # (lat,lon) and ij indexing: BL, BR, TL, TR
    # values_j = np.array([
    #     np.array([
    #     wnd_data['ws'].values[j, lat_sort_indices[i]-1, lon_sort_indices[i]-1],
    #     wnd_data['ws'].values[j, lat_sort_indices[i]-1, lon_sort_indices[i]]
    #     ]),
    #     np.array([
    #     wnd_data['ws'].values[j, lat_sort_indices[i], lon_sort_indices[i]-1],
    #     wnd_data['ws'].values[j, lat_sort_indices[i], lon_sort_indices[i]],
    #     ])
    # ])

    # bilinear_value_j = scipy.interpolate.interpn(points, values_j, x_i, method='linear')

    # # Proof plot:
    # fig, ax = plt.subplots()
    # LON, LAT = np.meshgrid(lon_model, lat_model)
    # ax.scatter(LON, LAT)
    # ax.scatter(x_i[1], x_i[0])
    # ax.set_ylim(x_i[0]-0.05, x_i[0]+0.05)
    # ax.set_xlim(x_i[1]-0.05, x_i[1]+0.05)
    # ax.annotate(round(bilinear_value_j[0],2),
    #             (x_i[1], x_i[0]))
    # ax.annotate(round(values_j[0,0],2),
    #             (points[1][0], points[0][0])) # bottom left
    # ax.annotate(round(values_j[0,1],2),
    #             (points[1][1], points[0][0])) # bottom right
    # ax.annotate(round(values_j[1,0],2),
    #             (points[1][0], points[0][1])) # top left
    # ax.annotate(round(values_j[1,1],2),
    #             (points[1][1], points[0][1])) # top right
    # ####
    # #TODO: put into a unit test:
    # x_top = np.interp(x_i[1], (points[1][0], points[1][1]), (values_j[1,0], values_j[1,1]))
    # x_bot = np.interp(x_i[1], (points[1][0], points[1][1]), (values_j[0,0], values_j[0,1]))
    # y = np.interp(x_i[0], (points[0][0], points[0][1]), (x_bot, x_top))
