"""TODO module docstr"""

__all__ = ['haversine_distance']

import numpy as np


def eulerian_dist(x1,y1,x2,y2):
    """
    Compute the Eulerian distance b/t two points

    """ 
    dx = np.subtract(x2,x1)
    dy = np.subtract(y2,y1)
    dist=np.sqrt(np.square(dx)+np.square(dy))
    return dist


def haversine_distance(lon1, lat1, lon2, lat2):
    """ 
    Calculate the great circle distance and bearing (wrt point 1) between two points on the earth 
    https://gis.stackexchange.com/questions/61924/python-gdal-degrees-to-meters-without-reprojecting 
    https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/gc_simple.pdf
    
    """

    # radius of the Earth
    r_earth = 6378.137 # 6371 km

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distkm = r_earth * c

    # bearing
    S = np.cos(lat2)*np.sin(-dlon)
    C = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = (-np.degrees(np.arctan2(S,C))) % 360

    # validation from R. Bullock: dist,bearing = haversine_distance(105.2833,40.0167,-137.65,-33.9333)
    return distkm, bearing
