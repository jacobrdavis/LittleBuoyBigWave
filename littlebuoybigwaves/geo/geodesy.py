"""Basic geographic calculations on the spherical earth.

A summary of the basic equations can be found here:

    http://www.movable-type.co.uk/scripts/latlong.html

For a proper geodesic algorithms and implemenations, see:

Karney, C.F.F. Algorithms for geodesics. J Geod 87, 43-55 (2013).
    https://doi.org/10.1007/s00190-012-0578-z

GeographicLib - https://geographiclib.sourceforge.io/index.html

TODO:
- update `haversine_distance_pairwise` function variable names
- update `euclidean_dist`
- docstr for `destination_coordinates`
"""

__all__ = [
    "euclidean_distance",
    "haversine_distance",
    "haversine_distance_pairwise",
    "great_circle_pathwise",
    "great_circle_pairwise",
    "destination_coordinates",
    "reciprocal_bearing",
    "get_extent",
]

from typing import Tuple

import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    """
    Compute the euclidean distance b/t two points

    """
    dx = np.subtract(x2, x1)
    dy = np.subtract(y2, y1)
    return np.sqrt(np.square(dx)+np.square(dy))


def haversine_distance(longitude, latitude, **kwargs):
    """ Alias to renamed function `great_circle_pathwise` """
    return great_circle_pathwise(longitude, latitude, **kwargs)


def great_circle_pathwise(
    longitude: np.ndarray,
    latitude: np.ndarray,
    earth_radius: float = 6378.137,
    mod_bearing: bool = True
) -> Tuple:
    """
    Computes the great circle distance (km) and true fore bearing (deg) along a
    path using adjacent values in `longitude` and `latitude`.

    For two longitude and latitude pairs, the great circle distance is the
    shortest distance between the two points along the Earth's surface. This
    distance is calculated using the Haversine formula. The first instance in
    longitude and latitude is designated as point `a`; the second instance is
    point `b`. The true fore bearing is the bearing, measured from true north,
    of `b` as seen from `a`.

    Note:
        When given `latitude` and `longitude` of shape (n,), n > 1, the great
        circle distance and fore bearing will be calculated between adjacent
        entries such that the returned arrays will be of shape (n-1,). To
        compute the great circle distance and bearings for distinct pairs of
        coordinates, use `haversine_distance_pairwise`.

    Args:
        longitude (np.array): of shape (n,) in units of decimal degrees
        latitude (np.array): of shape (n,) in units of decimal degrees
        earth_radius (float, optional): earth's radius in units of km. Defaults to 6378.137 km (WGS-84)
        mod_bearing (bool, optional): return bearings modulo 360 deg. Defaults to True.

    Raises:
        ValueError: if longitude or latitude are less than size of 2.

    Returns:
        Tuple[np.array, np.array]: great circle distances (in km) and true fore
        bearings between adjacent longitude and latitude pairs; shape (n-1,)

    Example: A trajectory along the Earth's equator.
    ```
    >> longitude = np.array([0, 1, 2, 3])
    >> latitude = np.array([0, 0, 0, 0])
    >> distance_km, bearing_deg = haversine_distance(longitude, latitude)
    >> distance_km
        array([111.19, 111.15, 111.08])  # 111 km ~ 60 nm
    >> bearing_deg
        array([90., 90., 90.]))
    ```
    """
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    if longitude.size <= 1 or latitude.size <= 1:
        raise ValueError("`longitude` and `latitude` must have size"
                         " of at least 2.")

    # Offset the longitude and latitude by one index to compute the haversine
    # distance and bearing between adjacent positions.
    longitude_a = longitude[0:-1]
    longitude_b = longitude[1:]
    latitude_a = latitude[0:-1]
    latitude_b = latitude[1:]

    #TODO: everything past here should really be put into the pairwise fn!
    # Convert decimal degrees to radians
    longitude_a_rad, latitude_a_rad = map(np.radians, [longitude_a, latitude_a])
    longitude_b_rad, latitude_b_rad = map(np.radians, [longitude_b, latitude_b])

    # Difference longitude and latitude
    longitude_difference = longitude_b_rad - longitude_a_rad
    latitude_difference = latitude_b_rad - latitude_a_rad

    # Haversine formula
    a_1 = np.sin(latitude_difference / 2) ** 2
    a_2 = np.cos(latitude_a_rad)
    a_3 = np.cos(latitude_b_rad)
    a_4 = np.sin(longitude_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a_1 + a_2 * a_3 * a_4))
    distance_km = earth_radius * c

    # True bearing
    bearing_num = np.cos(latitude_b_rad) * np.sin(-longitude_difference)
    bearing_den_1 = np.cos(latitude_a_rad) * np.sin(latitude_b_rad)
    bearing_den_2 = - np.sin(latitude_a_rad) * np.cos(latitude_b_rad) * np.cos(longitude_difference)
    bearing_deg = -np.degrees(np.arctan2(bearing_num, bearing_den_1 + bearing_den_2))

    if mod_bearing:
        bearing_deg = bearing_deg % 360

    return distance_km, bearing_deg


def haversine_distance_pairwise(longitude_a, latitude_a, longitude_b, latitude_b, **kwargs):
    """ Alias to renamed function `great_circle_pairwise` """
    return great_circle_pairwise(longitude_a, latitude_a, longitude_b, latitude_b, **kwargs)


def great_circle_pairwise(
    longitude_a: np.ndarray,
    latitude_a: np.ndarray,
    longitude_b: np.ndarray,
    latitude_b: np.ndarray,
    earth_radius: float = 6378.137,
    mod_bearing: bool = True
) -> Tuple:
    """
    Computes the great circle distance (km) and true fore bearing (deg) between
    pairs of observations in input arrays `longitude_a` and `longitude_b` and
    `latitude_a` and `latitude_b`.

    For two longitude and latitude pairs, the great circle distance is the
    shortest distance between the two points along the Earth's surface. This
    distance is calculated using the Haversine formula. The instances in
    `longitude_a` and `latitude_a` are designated as point `a`; the instances
    in `longitude_b` and `latitude_b` then form point `b`. The true fore
    bearing is the bearing, measured from true north, of `b` as seen from `a`.

    Note:
        When given `latitude_a/b` and `longitude_a/b` of shape (n,), n > 1,
        the great circle distance and fore bearing will be calculated between
        `a` and `b` entries such that the returned arrays will be of shape
        (n,). To compute the great circle distance and bearings between
        adjacent coordinates of single longitude and latitude arrays (i.e.,
        along a trajectory), use `great_circle_pathwise`.

    Args:
        longitude_a (np.array): of shape (n,) in units of decimal degrees
        latitude (np.array): of shape (n,) in units of decimal degrees
        earth_radius (float, optional): earth's radius in units of km. Defaults to 6378.137 km (WGS-84)
        mod_bearing (bool, optional): return bearings modulo 360 deg. Defaults to True.

    Returns:
        Tuple[np.array, np.array]: great circle distances (in km) and true fore
        bearings between adjacent longitude and latitude pairs; shape (n,)

    Example: A trajectory along the Earth's equator.
    ```
    >> #TODO:
    ```
    """

    # convert decimal degrees to radians
    longitude_a_rad, latitude_a_rad = map(np.radians, [longitude_a, latitude_a])
    longitude_b_rad, latitude_b_rad = map(np.radians, [longitude_b, latitude_b])

    # haversine formula
    dlon = longitude_b_rad - longitude_a_rad
    dlat = latitude_b_rad - latitude_a_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_a_rad) * np.cos(latitude_b_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    distance_km = earth_radius * c

    # bearing
    S = np.cos(latitude_b_rad)*np.sin(-dlon)
    C = np.cos(latitude_a_rad)*np.sin(latitude_b_rad) - np.sin(latitude_a_rad)*np.cos(latitude_b_rad)*np.cos(dlon)
    bearing_deg = (-np.degrees(np.arctan2(S, C)))
    if mod_bearing:
        bearing_deg = bearing_deg % 360

    return distance_km, bearing_deg


def destination_coordinates(
    origin_longitude: np.array,
    origin_latitude: np.array,
    distance: float,
    bearing: float,
    earth_radius: float = 6378.137
) -> Tuple:
    # TODO: docstr

    # Convert decimal degrees to radians
    origin_longitude_rad = np.radians(origin_longitude)
    origin_latitude_rad = np.radians(origin_latitude)

    bearing_rad = np.radians(bearing)
    angular_distance = distance/earth_radius

    # Calculate the latitude at the destination
    lat_term_1 = np.sin(origin_latitude_rad) * np.cos(angular_distance)
    lat_term_2 = np.cos(origin_latitude_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    destination_latitude_rad = np.arcsin(lat_term_1 + lat_term_2)

    # Calculate the longitude at the destination
    lon_term_1 = np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(origin_latitude_rad)
    lon_term_2 = np.cos(angular_distance)
    lon_term_3 = np.sin(origin_latitude_rad) * np.sin(destination_latitude_rad)
    lon_arctan = np.arctan2(lon_term_1, lon_term_2 + lon_term_3)
    destination_longitude_rad = origin_longitude_rad + lon_arctan

    # Return as tuple coordinate in decimal degrees
    destination = (np.rad2deg(destination_longitude_rad.squeeze()),
                   np.rad2deg(destination_latitude_rad.squeeze()))
    # destination = (np.rad2deg(destination_longitude_rad.item()),
    #                np.rad2deg(destination_latitude_rad.item()))

    return destination


def reciprocal_bearing(bearing):
    """Return the reciprocal (back) bearing of the input bearing.

    Args:
        bearing (float | np.array): bearing in decimal degrees.

    Returns:
        float | np.array: the back bearing(s) (modulo 360 degrees).
    """
    return (bearing + 180) % 360


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