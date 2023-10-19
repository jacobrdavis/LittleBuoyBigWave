"""
A collection of functions for working with COAMPS-TC model data.

#TODO:
    - use np for datetimes?
    - test shapes in unittests
"""
__all__ = [
    'read_coamps_wnd_file',
]

import re
from datetime import datetime

import numpy as np

PARAMETER_REGEX = {
    'DT': r'DT=\d{10}',
    'iLat': r'iLat=\s*\d*',
    'iLong': r'iLong=\s*\d*',
    'DX': r'DX=\s*\d*\.*\d*',
    'DY': r'DY=\s*\d*\.*\d*',
    'SWLat': r'SWLat=\s*-?\d+\.*\d*',
    'SWLon': r'SWLon=\s*-?\d+\.*\d*',
}


def read_coamps_wnd_file(wnd_file: str, uniform_parameters: bool = False): #
    """
    Read an Oceanweather WIN file (.wnd) output by COAMPS.

    More information on this format can be found here:
    - https://coast.nd.edu/reports_papers/SELA_2007_IDS_2_FinalDraft/App
      %20D%20PBL-C%20WIN_PRE%20File%20Format.pdf
    - https://wiki.adcirc.org/NWS12

    Args:
        wnd_file (str): path to the .wnd file

        uniform_parameters (bool): flag specifying if the header
            parameters are uniform or not. If True, parameters are only
            read from the first header. Otherwise, parameters are re-
            read for every header.

    Raises:
        ValueError: if there is an unexpected number of values between
            headers based on the preceeding header parameters.

    Returns:
        dict: containing wind coordinates and fields.
    """
    parameters = {}
    data = []
    data_count = 0

    fields = {
        'u': [],
        'v': [],
    }
    coords = {
        'time': [],
        'longitude': [],
        'latitude': [],
    }
    with open(wnd_file, mode='r', encoding='utf-8') as file:
        for line in file:
            # Evaluate the line to see if it is a header. If so, check
            # to see if any data lines have been read and parse them.
            # Then parse the header parameters (if it is the first pass
            # or if the they are not uniform) and reset the data fields.
            line_is_header = re.search(PARAMETER_REGEX['DT'], line)
            if line_is_header:
                if len(data) > 0 and data_count != 2*n_val:
                    raise ValueError(
                        f'Unexpected number of values between headers; '
                        f'got {data_count} values but expected {2*n_val}.')
                
                coords['time'].append(_parse_time(line_is_header))

                if not parameters or not uniform_parameters:
                    parameters = _parse_header(line)
                    # coords['time'].append(parameters['DT'])
                    coords['longitude'].append(_construct_grid_coord(
                                                        parameters['SWLon'],
                                                        parameters['DX'],
                                                        parameters['iLong']))
                    coords['latitude'].append(_construct_grid_coord(
                                                        parameters['SWLat'],
                                                        parameters['DY'],
                                                        parameters['iLat']))
                    n_x = parameters['iLong']
                    n_y = parameters['iLat']
                    n_val = n_x * n_y

                data = []
                data_count = 0

            # If not a header line and parameters exist (if a header has
            # been read previously) collect data until the next header.
            elif parameters:
                line_as_floats = np.array(line.replace('\n', '').split(),
                                            dtype=float)
                data.append(line_as_floats)
                data_count = data_count + len(line_as_floats)

                if data_count == 2*n_val:
                    data = np.concatenate(data)
                    fields['u'].append(data[0:n_val].reshape(n_y, n_x))
                    fields['v'].append(data[n_val:2*n_val].reshape(n_y, n_x))
            else:
                continue

    # Convert the coordinates and fields to numpy arrays and return.
    # The fields are indexed as [time][latitude][longitude]
    coords['time'] = np.array(coords['time'])
    coords['latitude'] = np.squeeze(np.array(coords['latitude']))
    coords['longitude'] = np.squeeze(np.array(coords['longitude']))
    fields['u'] = np.moveaxis(np.dstack(fields['u']), 2, 0)
    fields['v'] = np.moveaxis(np.dstack(fields['v']), 2, 0)
    fields['ws'] = np.sqrt(fields['u']**2 + fields['v']**2)

    return coords, fields


def _parse_time(time):
    name, value = _parse_parameter(time.group())
    if name == 'DT':
        value = datetime.strptime(value, '%Y%m%d%H')
    else:
        raise ValueError(f'Incorrect time parameter `{name}`.')
    return value


def _parse_header(header):
    """
    Parse a .wnd file header.

    Args:
        header (str): .wnd file header

    Raises:
        ValueError: if parameter is not 'DT', 'iLat', 'iLong', 'DX',
        'DY', 'SWLat', or 'SWLon'.

    Returns:
        dict: parameters as parsed from the header
    """
    parameters = {}
    for param, regex in PARAMETER_REGEX.items():
        match = re.search(regex, header)
        if match:
            name, value = _parse_parameter(match.group())

        if param == 'DT':
            pass
        #     value = datetime.strptime(value, '%Y%m%d%H')
        elif param in ['iLat', 'iLong']: #elif
            value = int(value)
        elif param in ['DX', 'DY', 'SWLat', 'SWLon']:
            value = float(value)
        else:
            raise ValueError(f'Unknown parameter `{param}`.')
        parameters[name] = value

    return parameters


def _parse_parameter(parameter_string):
    """
    Parse .wnd header parameter into its name and value.

    Args:
        parameter_string (str): name and value, separated by '='

    Returns:
        (str, str): parameter name and value
    """
    parameter_string = parameter_string.split('=')
    name = parameter_string[0].strip() # see also .partition
    value = parameter_string[1].strip()
    return name, value


def _construct_grid_coord(start, step, n_steps):
    """
    Construct a .wnd file grid coordinate (latitude or longitude) from
    its start, step, and number of steps (as specified in the header
    parameters).

    Args:
        start (int): start of grid coordinate, typ. SW from the corner
        step (int): grid coordinate step size
        n_steps (int): number of steps in the grid coordinate direction

    Returns:
        ndarray: constructed grid coordinate
    """
    # np.arange(start, start + n_steps*step, step)
    return np.linspace(start, start + (n_steps-1)*step, num=n_steps)
