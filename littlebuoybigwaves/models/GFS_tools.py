"""
Author: @jacobrdavis

A collection of functions for working with NOAA Global Forecast System (GFS) model data. Includes
tools for querying the GFS S3 bucket hosted on Amazon Web Services (AWS). For more information, 
see: https://registry.opendata.aws/noaa-gfs-bdp-pds/. 
User interface here: https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html

Contents:
    - download_dir(prefix, local, bucket, client, keymatch)
    - download_file(bucketfile,local,bucket,client):
    - generate_daterange(start,end):
    - querybydate(daterange,model,submodel,type,region,resolution,forecasthour,bucket,localfolderbase):
    - collateGFSwave(startDate, endDate, path, model, submodel, region, resolution, forecasthour)
    - main()

Log:
    - 2021-11-11, J.Davis: created NOAA_GFS_AWS_query_tools.py
    - 2022-01-07, J.Davis: updated for hurricane wave slope comparisons
    - 2022-09-06, J.Davis: renamed to GFS_tools and updated docstrings
    - 2022-09-07 added collateGFSwave(...) and updated to store every key in GFS dataset

TODO:
    - implement a model dict object that can be passed to each function w/o need for input every time
"""
#%%
import pandas as pd
import boto3 #https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
from botocore import UNSIGNED
from botocore.config import Config
import os
from datetime import datetime
import datetime
from dateutil import parser
import warnings
import pickle
import xarray as xr
import numpy as np
from typing import List

#%% functions
def download_dir(prefix: str, local: str, bucket: str, client: boto3.client, keymatch: str):
    """
    Function to download all of the files in an S3 bucket 'directory'. Adapted from original 
    solution created by Grant Langseth at:
    https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
    
    Input:
        - prefix, pattern to match in S3 (e.g. 'gfs.20210821/00/wave/gridded')
        - local, local path to folder in which to place files
        - bucket, S3 bucket with target contents (e.g. 'noaa-gfs-bdp-pds')
        - client, initialized s3 client object (e.g. boto3.client('s3',...))
        - keymatch, pattern to match in each key or individual bucket file (e.g. 'global')
    
    Output:
        - target contents stored in directory specified by 'local'

    Example:

    For the GFS model, to download all of the global, 0.16 degree resolution gridded wave data at 
    the 0th hour, stored here: 
    https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html#gfs.20210821/00/wave/gridded/
    
    Use this functions as follows:
        download_dir(prefix = 'gfs.20210821/00/wave/gridded',
                    local = './', 
                    bucket = 'noaa-gfs-bdp-pds', 
                    client = boto3.client('s3', config=Config(signature_version=UNSIGNED)), 
                    keymatch = 'global.0p16')
    """
    keys = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        if results.get('Contents') is None: # if prefix is a file and not a folder
            keys = [prefix]
            next_token = None
        else: # prefix is a folder
            contents = results.get('Contents')
            for i in contents:
                k = i.get('Key')
                if k[-1] != '/': # ignore subdirectories (see original solution if directories are desired)
                    if keymatch in k:
                        keys.append(k)
            next_token = results.get('NextContinuationToken')
    for k in keys:
        filename = k.split('/')[-1]
        dest_pathname = os.path.join(local, filename)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


def download_file(bucketfile: str, local: str, bucket: str, client: str):
    """
    Function to download a single file from an S3 bucket.

    Input:
        - bucketfile, fullfile to match in S3
        - local, local path to folder in which to place file
        - bucket, S3 bucket with target contents
        - client, initialized S3 client object

    Output:
        - target file stored in directory specified by 'local'
    """
    filename = bucketfile.split('/')[-1]
    dest_pathname = os.path.join(local, filename)
    if not os.path.exists(os.path.dirname(dest_pathname)):
        os.makedirs(os.path.dirname(dest_pathname))
    client.download_file(bucket, bucketfile, dest_pathname)

def generate_daterange(start: pd.Timestamp,end: pd.Timestamp) -> List[str]: 
    """
    Helper function to create a date range from a start date to an end date and convert it to a
    list of strings for use in a GFS AWS query. Currently hard-coded for a timedelta of 1 day.

    Input:
        - start, start date as a datetime or pandas timestamp
        - end, end date as a datetime or pandas timestamp 
       
    Output:
        - daterangestr, a list of date strings with format %Y%m%d (yyyymmdd)
    """
    # init date range variables
    daterange = []
    daterangestr = ''

    # cast inputs as pandas timestamps, if they are not already
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # generate the date range, incrementing by day
    daterange = [start + datetime.timedelta(days=x) for x in range(0,(end-start).ceil("D").days+1)]

    # convert output to datestr in GFS AWS bucket format
    daterangestr = [date.strftime("%Y%m%d") for date in daterange]
    return daterangestr

def querybydate(daterange: List[str], model: str, submodel: str, type: str, region: str, 
                resolution: str, forecasthour: str, bucket: str, localfolderbase: str):
    """
    Function to query the GFS AWS S3 bucket by date in yyyymmdd format and by the target 
    model, submodel, model output type, region, resolution, and forecast hour.

    Input:
        - daterange: list of dates in 'yyyymmdd' format (e.g. created using generate_daterange)
        - model, name of the forecast model (e.g. 'gfs')
        - submodel, name of the submodel component (e.g. 'wave')
        - type, model output type (e.g. 'gridded')
        - region, model region (e.g. 'global')
        - resolution, model output resolution (e.g. '0p25')
        - forecasthour, forecast lead time (e.g. '000')
        - bucket, S3 bucket with target contents (e.g. 'noaa-gfs-bdp-pds')
        - localfolderbase,  local path to folder in which to place subfolders by date

    Output:
        - target contents stored in 'localfolderbase' directory, organized into subfolders by date
    
    Example:
    
    For the GFS model, to download all of the global, 0.25 degree resolution gridded wave data at 
    all forecast hours (0000, 0600, 1200 and 1800) with 0 hour lead time over week of September 4, 
    2022, use this functions as follows:
        querybydate(daterange = generate_daterange(start = datetime(2022,9,4), end = datetime(2022,9,11))
                    model = 'gfs',
                    submodel = 'wave',
                    type = 'gridded',
                    region = 'global',
                    resolution = '0p25',
                    forecasthour = '000',
                    bucket = 'noaa-gfs-bdp-pds',
                    localfolderbase = './')
    """
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED)) # https://github.com/boto/boto3/issues/1200
    
    print(f'Attempting to download {4*len(daterange)} files to {localfolderbase}.')
    for date in daterange:
        localfolder = f'{localfolderbase}/{date}'
        for hour in ['00','06','12','18']:
            foldername = f'{model}.{date}/{hour}/{submodel}/{type}'
            filename = f'{model}{submodel}.t{hour}z.{region}.{resolution}.f{forecasthour}.grib2'
            bucketfile = f'{foldername}/{filename}'
            download_file(bucketfile=bucketfile,local=localfolder,bucket=bucket,client=s3_client)

    print('Complete.')

def collateGFSwave(startDate: datetime.datetime, endDate: datetime.datetime, path: str, model: str,
                   submodel: str, region: str, resolution: str, forecasthour: str) -> List[dict]:
    """
    Function to read GFSWave files into an dictionary and collate those dictionaries into a list.
    
    Input:
        - startDate, start of date range to collate GFS data
        - endDate, end of date range to collate GFS data
        - path, directory containing GFSwave .grib files
            * assumes data is stored in subfolders labeled by date in yyyymmdd format
        - model, name of the forecast model (e.g. 'gfs')
        - submodel, name of the submodel component (e.g. 'wave')
        - region, model region (e.g. 'global')
        - resolution, model output resolution (e.g. '0p25')
        - forecasthour, forecast lead time (e.g. '000')

    Output:
        - data, list of dictionaries containing GFSwave values; sorted by increasing time
          corresponding to date and forecast hour

    Example:
    
        data = collateGFSwave(startDate=datetime(2022,9,6), 
                              endDate = =datetime(2022,9,11),
                              path = '../GFS_data/',
                              model = 'gfs',
                              submodel = 'wave',
                              region = 'global',
                              resolution = '0p25',
                              forecasthour = '000')
    """
    data = list()
    dateRange =  [startDate + datetime.timedelta(days=x) for x in range(0, (endDate-startDate).days)]
    #TODO: should this be used?
    # dateRange =  [start + timedelta(days=x) for x in range(0, (end-start).ceil("D").days+1)]

    for d in dateRange:
        date = d.strftime('%Y%m%d')
        for h in ['00','06','12','18']:
            gribfile= f"{path}{date}/{model}{submodel}.t{h}z.{region}.{resolution}.f{forecasthour}.grib2"
            print(f'Loading {gribfile}')
            #TODO: make fun
            dataset = xr.open_dataset(gribfile, engine='cfgrib') # NOTE: pynio may be faster engine
            varKeys = ['time','latitude','longitude'] + list(dataset.keys()) 
            data.append({key : dataset[key].data for key in varKeys})
            #TODO: make fun
            dataset.close()
    print('Done.')       
    return data

"""stand-alone and testing"""
def main():
    #%% input:
    bucket = 'noaa-gfs-bdp-pds'
    model = 'gfs'
    date    = '20210821' # YYYYMMDD
    hour = '00' # leave empty if all (00,06,12,18)
    submodel = 'wave' #'atmos' NOTE: gfs entries prior to gfs.20210322 do not appear to adhere to this directory structure
    type='gridded'
    resolution ='0p25' # 0p16
    region = 'global' #'atlocn'
    forecasthour = '000'
    
    foldername = f'{model}.{date}/{hour}/{submodel}/{type}'
    filename   = f'{model}{submodel}.t{hour}z.{region}.{resolution}.f{forecasthour}.grib2'
    bucketfile = f'{foldername}/{filename}'
    localfile  = './'

    #%% download by date range:
    start = datetime.datetime.strptime("2021-08-15", "%Y-%m-%d")
    end = datetime.datetime.strptime("2021-08-30", "%Y-%m-%d")
    daterange = generate_daterange(start,end)
    querybydate(daterange,model,submodel,type,region,resolution,forecasthour,bucket,localfolderbase='./testquerybydate')

    #%% download entire directory:
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED)) # https://github.com/boto/boto3/issues/1200
    download_dir(prefix=foldername, local=localfile, bucket=bucket, client=s3_client,keymatch=region)

if __name__=='__main__':
    main()
