"""
Unit tests for the models sub-package.

TODO:
    - x
"""

import os
import unittest
from datetime import datetime
from unittest.mock import patch, mock_open

import numpy as np

from littlebuoybigwaves.models import gfs_tools as gfs
from littlebuoybigwaves.models import era5_tools as era5
from littlebuoybigwaves.models import coamps_tools as coamps

# class TestGfsTools(unittest.TestCase):
#     """ Unit tests for gfs tools submodule """
#     print('TODO')


# class TestEra5Tools(unittest.TestCase):
#     """ Unit tests for era5 tools submodule """
#     print('TODO')


class TestCoampsTools(unittest.TestCase):
    """ Unit tests for coamps tools submodule """

    def setUp(self):
        """ Sample wnd data for testing """
        self.wnd_data = (
        "Oceanweather WIN/PRE Format                        2022092400     2022100600\n"
        "iLat= 4iLong= 4DX=0.2000DY=0.2000SWLat= 5.00000SWLon=-100.000DT=202209240000\n"
        "0.1514    0.1462    0.1440    0.1613    0.1823    0.2128    0.2314    0.2314\n"
        "0.2125    0.1764    0.1569    0.1472    0.1896    0.2472    0.3502    0.4590\n"
        "1.9472    1.9851    2.0417    2.1100    2.2236    2.3550    2.5240    2.7003\n"
        "2.8957    3.0919    3.2837    3.4670    3.6189    3.7646    3.8986    4.0547\n"
        "iLat= 4iLong= 4DX=0.2000DY=0.2000SWLat= 5.00000SWLon=-100.000DT=202209240100\n"
        "0.1514    0.1462    0.1440    0.1613    0.1823    0.2128    0.2314    0.2314\n"
        "0.2125    0.1764    0.1569    0.1472    0.1896    0.2472    0.3502    0.4590\n"
        "1.9472    1.9851    2.0417    2.1100    2.2236    2.3550    2.5240    2.7003\n"
        "2.8957    3.0919    3.2837    3.4670    3.6189    3.7646    3.8986    4.0547\n"
        )

        self.time_coord = np.array([
            datetime(2022, 9, 24, 0, 0),
            datetime(2022, 9, 24, 1, 0)
        ])

        self.longitude_coord = np.array(
            [-100. ,  -99.8,  -99.6,  -99.4],
        )

        self.latitude_coord = np.array(
            [5. , 5.2, 5.4, 5.6],
        )

        self.u_field = np.array([
            [0.1514, 0.1462, 0.144 , 0.1613],
            [0.1823, 0.2128, 0.2314, 0.2314],
            [0.2125, 0.1764, 0.1569, 0.1472],
            [0.1896, 0.2472, 0.3502, 0.459 ]
        ])


        self.v_field = np.array([
            [1.9472, 1.9851, 2.0417, 2.11  ],
            [2.2236, 2.355 , 2.524 , 2.7003],
            [2.8957, 3.0919, 3.2837, 3.467 ],
            [3.6189, 3.7646, 3.8986, 4.0547]
        ])

        self.ws_field = np.array([
            [1.95307701, 1.99047644, 2.04677182, 2.11615635],
            [2.23106034, 2.36459486, 2.53458517, 2.71019668],
            [2.90348665, 3.09692792, 3.28744632, 3.47012346],
            [3.62386332, 3.77270738, 3.91429713, 4.08059715],
        ])

    def test_read_coamps_wnd_file_nonuniform(self):
        """
        Read in test wnd data with uniform parameters set to False. The
        output latitude and longitude coordinates will have the same
        length as the time coordinate.
        """
        with patch("builtins.open",\
                              mock_open(read_data=self.wnd_data)) as mock_file:
            
            coords, fields = coamps.read_coamps_wnd_file(
                mock_file,
                uniform_parameters=False
            )

        np.testing.assert_allclose(
            np.array(coords['time'], dtype='datetime64').astype("float"),
            np.array(self.time_coord, dtype='datetime64').astype("float")
        )

        np.testing.assert_allclose(
            coords['latitude'],
            np.tile(self.latitude_coord, (2,1))
        )

        np.testing.assert_allclose(
            coords['longitude'],
            np.tile(self.longitude_coord, (2,1))
        )

        np.testing.assert_allclose(
            fields['u'],
            np.tile(self.u_field, (2,1,1))
        )

        np.testing.assert_allclose(
            fields['v'],
            np.tile(self.v_field, (2,1,1))
        )

        np.testing.assert_allclose(
            fields['ws'],
            np.tile(self.ws_field, (2,1,1))
        )

    def test_read_coamps_wnd_file_uniform(self):
        """
        Read in test wnd data with uniform parameters set to True. The
        output latitude and longitude coordinates will have the shapes
        (iLong,) and (iLat,).
        """
        with patch("builtins.open",\
                              mock_open(read_data=self.wnd_data)) as mock_file:
            
            coords, fields = coamps.read_coamps_wnd_file(
                mock_file,
                uniform_parameters=True
            )

        np.testing.assert_allclose(
            np.array(coords['time'], dtype='datetime64').astype("float"),
            np.array(self.time_coord, dtype='datetime64').astype("float")
        )

        np.testing.assert_allclose(
            coords['latitude'],
            self.latitude_coord
        )

        np.testing.assert_allclose(
            coords['longitude'],
            self.longitude_coord
        )

        np.testing.assert_allclose(
            fields['u'],
            np.tile(self.u_field, (2,1,1))
        )

        np.testing.assert_allclose(
            fields['v'],
            np.tile(self.v_field, (2,1,1))
        )

        np.testing.assert_allclose(
            fields['ws'],
            np.tile(self.ws_field, (2,1,1))
        )
if __name__ == "__main__":
     unittest.main()