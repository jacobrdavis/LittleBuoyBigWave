"""
SWAN model functions.
"""

# TODO:
# - use config vars in Dataset creation

__all__ = [
    "read_swan_spc2d",
    "spc2d_to_xarray",
]


import numpy as np
import pandas as pd
import xarray as xr


def _parse_header(line):
    """ Parse line as a header and associated description. """
    line_split = line.split()
    header = line_split[0]
    description = ' '.join(line_split[1:])
    return header, description


def _parse_num_values(line):
    """ Parse line as an integer representing the number of values. """
    return int(line.split()[0])


def _parse_line_as_array(line):
    """ Parse line as an array of multiple floats.  """
    values = line.strip().split()
    return np.array([float(value) for value in values])


def _parse_line_as_float(line):
    """ Parse line as a single float. """
    return float(line.strip())


def _parse_array(lines):
    """ Parse lines as a 1D array of floats (one float per line). """
    return np.array([_parse_line_as_float(line) for line in lines])


def _parse_2d_array(lines):
    """ Parse lines as a 2D array of floats (multiple floats per line). """
    return np.array([_parse_line_as_array(line) for line in lines])


def _parse_datetime(line):
    """ Parse line as a single %Y%m%d.%H%M%S formatted datetime. """
    return pd.to_datetime(line.split()[0], format='%Y%m%d.%H%M%S')


def read_swan_spc2d(file_path: str) -> dict:
    """Read a SWAN spc2d file into memory as a dictionary.

    Parses longitude and latitude (LONLAT), absolute frequencies (AFREQ),
    nautical directions (NDIR), and directional energy density (EnDens).

    Args:
        file_path (str): Absolute or relative file path to the spc2d file.

    Returns:
        dict: SWAN spc2d data values and descriptions, keyed by header.
    """
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        lines = file.readlines()  # read entire file into memory (small)
        # Search for headers line-by-line and parse accordingly.
        for count, line in enumerate(lines):
            if 'LONLAT' in line:
                header, description = _parse_header(line)
                n_lonlat = _parse_num_values(lines[count + 1])
                values = _parse_2d_array(lines[count + 2: count + 2 + n_lonlat])
            elif 'AFREQ' in line:
                header, description = _parse_header(line)
                n_freq = _parse_num_values(lines[count + 1])
                values = _parse_array(lines[count + 2: count + 2 + n_freq])
            elif 'NDIR' in line:
                header, description = _parse_header(line)
                n_dir = _parse_num_values(lines[count + 1])
                values = _parse_array(lines[count + 2: count + 2 + n_dir])
            elif 'QUANT' in line:
                pass  # TODO: not currently handled
            elif 'EnDens' in line:
                header, description = _parse_header(line)
                datetime = _parse_datetime(lines[count + 3])
                factor = float(lines[count + 5])
                values = _parse_2d_array(lines[count + 6: count + 6 + n_freq])
                values = values * factor
                # Assign datetime to the data dictionary:
                data['datetime'] = {'values': [datetime], 'description': None}
            else:
                header = None
                values = None
                description = None

            # Assign values and description to the data dict by header:
            if header is not None:
                data[header] = {'values': values, 'description': description}

    return data


def spc2d_to_xarray(spc2d_dict: dict) -> xr.Dataset:
    """Create Dataset from spc2d dictionary as read by `read_swan_spc2d`. """
    # Create core dataset from the directional energy density spectrum.
    ds = xr.Dataset(
        coords={
            'frequency': (  # TODO: use config var namespace here
                'frequency',
                spc2d_dict['AFREQ']['values'],
                {'description': spc2d_dict['AFREQ']['description']},
            ),
            'direction': (
                'direction',
                spc2d_dict['NDIR']['values'],
                {'description': spc2d_dict['NDIR']['description']},
            ),
        },
        data_vars={
            'frequency_direction_energy_density': (
                ('frequency', 'direction'),
                spc2d_dict['EnDens']['values'],
                {'description': spc2d_dict['EnDens']['description']},
            ),
        }
    )
    # Add time to dimensions.
    ds = ds.expand_dims({'time': spc2d_dict['datetime']['values']})

    # Add longitude and latitude to the variables.
    ds['longitude'] = (
        'time',
        spc2d_dict['LONLAT']['values'][:, 0],
        {'description': spc2d_dict['LONLAT']['description']},
    )
    ds['latitude'] = (
        'time',
        spc2d_dict['LONLAT']['values'][:, 1],
        {'description': spc2d_dict['LONLAT']['description']},
    )

    return ds
