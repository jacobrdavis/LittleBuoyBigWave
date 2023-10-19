""" 
Unit tests for the geo sub-package.
"""

import unittest

import numpy as np

from littlebuoybigwaves.geo import geodesy as geo


class TestGeographyTools(unittest.TestCase):
    """ Unit tests for geography tools submodule """

    # def setUp(self):

    def test_haversine_distance_size_error(self):
        """
        Check an error is raised for haversize distance size.
        """
        longitude = np.array([0])
        latitude = np.array([0])

        with self.assertRaises(ValueError):
            geo.haversine_distance(longitude, latitude)

    def test_haversine_distance_list_to_numpy(self):
        """
        Check haversize distance can handle lists.
        """
        # Trajectory along the equator
        longitude = [0, 1, 2, 3]
        latitude = [0, 0, 0, 0]
        expected_distance = 111.3195 * np.ones(len(longitude)-1)
        expected_bearing = 90 * np.ones(len(longitude)-1)

        distance, bearing = geo.haversine_distance(longitude, latitude)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-05)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-05)

    def test_haversine_distance_equator(self):
        """
        Check haversize distance function along an equator trajectory.
        """
        # Trajectory along the equator
        longitude = np.arange(-180, 180, 1)
        latitude = np.zeros(longitude.shape)
        expected_distance = 111.3195 * np.ones(len(longitude)-1)  # 60 n mi
        expected_bearing = 90 * np.ones(len(longitude)-1)

        distance, bearing = geo.haversine_distance(longitude, latitude)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-05)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-05)

    def test_haversine_distance_prime_meridian(self):
        """
        Check haversize distance function along a prime meridan trajectory.
        """
        # Trajectory along the prime meridian
        latitude = np.arange(-90, 90, 1)
        longitude = np.zeros(latitude.shape)

        expected_distance = 111.3195 * np.ones(len(longitude)-1)  # 60 n mi
        expected_bearing = 0 * np.ones(len(longitude)-1)

        distance, bearing = geo.haversine_distance(longitude, latitude)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-05)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-05)

    def test_haversine_distance_intl_dateline(self):
        """
        Check haversize distance function along a southerly international
        dateline trajectory.
        """
        # Trajectory along the international dateline
        latitude = np.flip(np.arange(-90, 90, 1))
        longitude = 180 * np.ones(latitude.shape)

        expected_distance = 111.3195 * np.ones(len(longitude)-1)  # 60 n mi
        expected_bearing = 180 * np.ones(len(longitude)-1)

        distance, bearing = geo.haversine_distance(longitude, latitude)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-05)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-05)

    def test_haversine_distance_brest_to_boston(self):
        """
        Check haversize distance function from Brest to Boston
        """

        # Trajectory from Brest, France (48˚22'30"N, 4˚33'00"W) to
        # Boston, MA (42°21'40" N, 71°3'26"W).
        latitude = np.array([48.39, 42.37])
        longitude = np.array([-4.55, -71.06])

        expected_distance = 5090  # km
        expected_bearing = 288 * np.ones(len(longitude)-1)  # deg

        distance, bearing = geo.haversine_distance(longitude, latitude)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-02)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-02)

    def test_haversine_distance_pairwise_prime_meridian(self):
        """
        Check haversize distance function along a prime meridan trajectory.
        """
        # Trajectory along the prime meridian
        latitude_a = np.arange(-90, 90, 1)
        longitude_a = np.zeros(latitude_a.shape)
        latitude_b = np.zeros(latitude_a.shape)
        longitude_b = np.zeros(latitude_a.shape)

        expected_distance = 111.3195 * np.abs(latitude_a)  # 60 n mi
        expected_bearing = np.zeros(len(longitude_a))

        expected_bearing[int(len(longitude_a)/2)+1:] = 180

        distance, bearing = geo.haversine_distance_pairwise(longitude_a,
                                                            latitude_a,
                                                            longitude_b,
                                                            latitude_b)

        np.testing.assert_allclose(expected_distance, distance, rtol=1e-05)
        np.testing.assert_allclose(expected_bearing, bearing, rtol=1e-05)



if __name__ == "__main__":
     unittest.main()