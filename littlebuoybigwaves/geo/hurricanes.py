""" Hurricane-related functions.

"""

__all__ = [
    "query_nhc_api",
    "get_latest_nhc_kml_files",
    "read_shp_file",
    "read_kml_file",
    "set_best_track_datetime_index",
    "best_track_pts_to_intensity",
]


import fnmatch
import requests
from io import BytesIO
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

NHC_STORM_GRAPHICS_API = 'https://www.nhc.noaa.gov/storm_graphics/api/'
SAFFIR_SIMPSON = {
    'D' : {'range': (0  , 38.99),  'int': -1},
    'TS': {'range': (39 , 73.99),  'int':  0},
    '1' : {'range': (74 , 95.99),  'int':  1},
    '2' : {'range': (96 , 110.99), 'int':  2},
    '3' : {'range': (111, 129.99), 'int':  3},
    '4' : {'range': (130, 156.99), 'int':  4},
    '5' : {'range': (157, 200),    'int':  5},
}


def query_nhc_api(kml_file):
    response = requests.get(NHC_STORM_GRAPHICS_API + kml_file, timeout=10)
    zip_content = ZipFile(BytesIO(response.content))

    kml_files = []
    for file in fnmatch.filter(zip_content.namelist(), '*.kml'):
        kml_files.append(zip_content.open(file))
    return kml_files


def get_latest_nhc_kml_files(storm_id: str):
    #TODO:
    track_kml = query_nhc_api(f'{storm_id}_TRACK_latest.kmz')
    track_gdf = read_kml_file(track_kml[0])

    cone_kml = query_nhc_api(f'{storm_id}_CONE_latest.kmz')
    cone_gdf = read_kml_file(cone_kml[0])

    initialradii_kml = query_nhc_api(f'{storm_id}_initialradii_latest.kmz')
    initialradii_gdf = read_kml_file(initialradii_kml[0])

    return track_gdf, cone_gdf, initialradii_gdf


def read_shp_file(
    path: str,
    crs: str = "EPSG:4326",
    index_by_datetime: bool = True
) -> gpd.GeoDataFrame:
    """ Read a shape file (.shp) into a GeoDataFrame.

    Read a shape file (.shp) into a GeoDataFrame and assign a  coordinate
    reference system (crs).  By default, the WGS84 (EPSG:4326) datum is used.

    Args:
        path (str): path to the shape file.
        crs (str, optional): cartopy coordinate reference system. Defaults to
            "EPSG:4326").
        index_by_datetime (bool, optional): if True, assign the datetime as the
            GeoDataFrame index. Defaults to True.

    Returns:
        gpd.GeoDataFrame: shape file data in the specified crs.
    """
    shp_gdf = gpd.read_file(path)
    if index_by_datetime:
        try:
            set_best_track_datetime_index(shp_gdf)
        except KeyError as error:
            print(f'Unable to set datetime index due to KeyError ({error}).')

    return shp_gdf.to_crs(crs)


def read_kml_file(path: str) -> gpd.GeoDataFrame:
    """ Read a KML file (.kml) into a GeoDataFrame.

    Read a KML file (.kml) into a GeoDataFrame.  The output coordinate
    reference system (crs) will be the same as what is specified in the KML.

    Args:
        path (str): path to the KML file.

    Returns:
        gpd.GeoDataFrame: KML file data.
    """
    # Enable fiona driver
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    return gpd.read_file(path, driver='KML')

#     cone_path = './al102023_014adv_CONE.kml'  #marineobs_by_pgm.kml'
#     track_path = './al102023_014adv_TRACK.kml'
#     initialradii_path = './AL102023_2023083003_initialradii.kml'

#     # Read file
#     cone_gdf = gpd.read_file(cone_path, driver='KML')
#     track_gdf = gpd.read_file(track_path, driver='KML')
#     initialradii_gdf = gpd.read_file(initialradii_path, driver='KML')


def set_best_track_datetime_index(best_track: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NHC best track timestamps to datetimes.

    Convert the time information in NHC best track shapefiles, stored in
    separate year, month, day, hour, and minute fields, into a unified
    datetime and set it as the index.

    Args:
        best_track (pd.DataFrame): best track as downloaded from NHC

    Returns:
        pd.DataFrame: DataFrame with unified datetime index
    """
    datetime_columns = ['YEAR', 'MONTH', 'DAY','HOUR','MINUTE']
    best_track['HOUR'] = best_track['HHMM'].str[:2]
    best_track['MINUTE'] = best_track['HHMM'].str[2:]
    best_track["datetime"] = pd.to_datetime(best_track[datetime_columns],
                                            utc=True)
    best_track.set_index('datetime', inplace=True)
    best_track.drop(datetime_columns + ['HHMM'], axis=1, inplace=True)
    return best_track


def mph_2_knots(mph):
    """ Helper function convert wind speeds from mph to knots. """
    return mph*0.868976


def best_track_pts_to_intensity(pts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Categorize best track points using  Saffir Simpson scale.

    Categorize the best track intensities (provided as wind speeds) using the
    Saffir Simpson scale. The input GeoDataFrame must have an 'INTENSITY'
    column. The output has two additional columns ('saffir_simpson_label' and
    'saffir_simpson_int') which are the Saffir Simpson scale labels
    ['D', 'TS', '1', ... '5'] and corresponding integers [-1, 0, 1, ... 5]
    (which are useful for colormapping).

    Args:
        pts_gdf (gpd.GeoDataFrame): NHC best track points

    Returns:
        gpd.GeoDataFrame: original GeoDataFrame with columns for Saffir Simpson
            label and intensity.
    """
    for cat, definition in SAFFIR_SIMPSON.items():
        range_kn = definition['range']
        in_range = pts_gdf['INTENSITY'].between(mph_2_knots(range_kn[0]),
                                                mph_2_knots(range_kn[1]))
        pts_gdf.loc[in_range, 'saffir_simpson_label'] = cat
        pts_gdf.loc[in_range, 'saffir_simpson_int'] = definition['int']
        return pts_gdf
