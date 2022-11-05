"""
title: gribfilemethods.py
description: collection of methods for working with grib files.
author: Jake Davis
created on: 2021-11-11
last updated: 2022-01-07
"""
#%%
import pandas as pd
import Nio # https://anaconda.org/conda-forge/pynio ; conda install -c conda-forge/label/cf202003 pynio
import os
from datetime import datetime
from dateutil import parser
import warnings
import pickle
import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
#%% functions

#%% setting ecccodes env vars
"""
https://github.com/ecmwf/cfgrib/issues/87
https://confluence.ecmwf.int/display/UDOC/Local+configuration+-+ecCodes+BUFR+FAQ
https://confluence.ecmwf.int/display/ECC
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars
"""
#%%
# gribfile = "gfs.t00z.pgrb2full.0p50.f000.grib2" # works?
# gribfile = "gfswave.t00z.global.0p25.f000.grib2" # does not work
# gribfile = "gfswave.t00z.atlocn.0p16.f004.grib2"
# gribfile - "gfswave.t00z.global.0p16.f000.grib2"
gribfile="./grib/gfswave.t00z.global.0p25.f001.grib2"
# gribfile = "gfswave.t00z.global.0p25.f006.grib2"
# gribfile ="gfswave.t00z.arctic.9km.f023.grib2"
# gribfile = "gfswave.t00z.epacif.0p16.f055.grib2"
#%%
f = Nio.open_file(gribfile)
ds = xr.open_dataset(gribfile, engine="pynio")
df = ds.to_dataframe()

#%%

data = xr.open_dataset(gribfile, engine='cfgrib')
#%%
fig,ax = plt.subplots()
ax.plot(data.u.data)
#%%
df = data.to_dataframe()
#%%
# import pygrib
# gr = pygrib.open(gribfile)
# for g in gr:
#     print(g)

import pygrib
#%%
gr = pygrib.open(gribfile)
grbmsgs = gr.read()
temp_vals = grbmsgs[0].values
lats, lons = grbmsgs[0].latlons()
vals = np.ma.filled(temp_vals, fill_value=0.0)
#%%
# import matplotlib.pyplot as plt
#%%
fig,ax = plt.subplots()
ax.plot(vals)

#%%
grb = gr.select(name='U component of wind')[0]
U = grb.values
#%%
# https://towardsdatascience.com/virtual-environments-104c62d48c54
# pip freeze > requirements.txt
# pip install -r requirements.txt
#%%
import time
import calendar
from calendar import timegm
import wget

epoch_zero_time = calendar.timegm(
    (2021, 11,14, 0, 0, 0))
forecast_hour = 0

spot_t = []
spot_h = []

mod_t = []
mod_h = []

for ind_counter in range(0, 4):
    current_time = epoch_zero_time + ind_counter * 3600
    model_time = epoch_zero_time + (ind_counter // 6) * 6 * 3600
    forecast_hour = (ind_counter % 6)

    new_time_struct = time.gmtime(model_time)
    date_string = f'{new_time_struct.tm_year}{str(new_time_struct.tm_mon).zfill(2)}{str(new_time_struct.tm_mday).zfill(2)}'
    hour_string = str(new_time_struct.tm_hour).zfill(2)

    file_base = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_string}/{hour_string}/wave/gridded/'
    file_name = f'gfswave.t{hour_string}z.global.0p25.f{str(forecast_hour).zfill(3)}.grib2'

    local_file = f'./{date_string}_{file_name}'

    url = os.path.join(file_base, file_name)
    if not os.path.exists(local_file):
        filename = wget.download(url=url, out=local_file)

    data = xr.open_dataset(local_file, engine='cfgrib')
#%% https://github.com/ecmwf/cfgrib/issues/18

ds = xr.open_dataset(gribfile)
print(ds.u.data)
# %%
