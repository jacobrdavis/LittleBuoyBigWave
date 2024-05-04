""" 
Unit tests for the waves sub-package.

TODO:
    - write unit tests for spectral_wave_tools
    - write unit tests for general_wave_tools

"""

import unittest

import numpy as np
import waves


class TestSpectralWaveTools(unittest.TestCase):
    """ Unit tests for spectral wave tools submodule """

    def setUp(self):
        self.sample_energy = np.array(
            [1.19930880e-02, 3.60038400e-02, 4.80215040e-02, 7.80288000e-02,
             5.16120576e-01, 3.00672614e+00, 1.31252060e+01, 2.38798356e+01,
             2.06630339e+01, 1.14808013e+01, 6.04348416e+00, 5.20327987e+00,
             4.42908672e+00, 2.73066394e+00, 2.47861248e+00, 2.31655834e+00,
             2.01648538e+00, 1.54838630e+00, 1.31432448e+00, 1.12828416e+00,
             9.00218880e-01, 7.32168192e-01, 5.94149376e-01, 4.80116736e-01,
             4.02087936e-01, 3.30080256e-01, 2.70065664e-01, 2.76062208e-01,
             2.52051456e-01, 1.62029568e-01, 1.38043392e-01, 1.32022272e-01,
             1.14032640e-01, 7.80288000e-02, 5.40180480e-02, 3.60038400e-02,
             3.00072960e-02, 2.40107520e-02]
        )
        self.sample_freq = np.array(
            [0.0293 , 0.03906, 0.04883, 0.05859, 0.06836, 0.07813, 0.08789,
             0.09766, 0.10742, 0.11719, 0.12695, 0.13672, 0.14648, 0.15625,
             0.16602, 0.17578, 0.18555, 0.19531, 0.20508, 0.21484, 0.22461,
             0.23438, 0.24414, 0.25391, 0.26367, 0.27344, 0.2832 , 0.29297,
             0.30273, 0.3125 , 0.32227, 0.33203, 0.35156, 0.38086, 0.41016,
             0.43945, 0.46875, 0.49805]
        )
        self.sample_spread = np.array(
            [60.482, 60.043, 58.489, 59.761, 49.501, 32.907, 27.017, 29.276,
             31.915, 36.077, 42.216, 38.31 , 38.798, 43.726, 42.237, 38.999,
             39.235, 43.452, 43.38 , 43.401, 46.813, 47.535, 48.714, 46.423,
             45.404, 46.414, 49.535, 51.433, 50.145, 52.748, 54.545, 54.202,
             54.85 , 58.5  , 61.566, 60.812, 62.073, 63.614]
        )
        self.sample_direction = np.array(
            [63.661, 69.837, 70.197, 67.172, 68.116, 66.766, 72.462, 75.214,
             79.938, 80.951, 75.945, 76.384, 72.507, 69.916, 68.67 , 64.935,
             66.283, 65.088, 65.296, 62.487, 58.806, 55.813, 59.352, 57.721,
             54.453, 45.589, 45.632, 55.269, 52.985, 47.269, 55.855, 55.888,
             52.551, 51.633, 55.648, 60.796, 52.208, 44.484]
        )

    # mean square slope function tests

    def test_mean_square_slope_length_1d(self):
        """
        Check for proper output length. For 1-dimensional energy and
        frequency arrays, the mean square slope is a single value.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq)
        self.assertEqual(mss.ndim, 0)
        self.assertEqual(bw.ndim, 0)
        self.assertEqual(np.shape(freq_range_logical),
                         np.shape(self.sample_freq))

    def test_mean_square_slope_type(self):
        """
        Check for proper output type.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq)
        self.assertIsInstance(mss, float)
        self.assertIsInstance(bw, float)
        self.assertIsInstance(freq_range_logical, np.ndarray)
        self.assertIsInstance(freq_range_logical[0], np.bool_)

    def test_mean_square_slope_total(self):
        """
        Check the total mss value for the sample arrays. Cross 
        validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm=None)
        self.assertAlmostEqual(mss, 0.0121, 4)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_total_fnorm(self):
        """
        Check the f-normalized total mss value for the sample arrays. 
        Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm='frequency')
        self.assertAlmostEqual(mss, 0.0259, 3)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_total_dnorm(self):
        """
        Check the dir-normalized total mss value for the sample arrays. 
        Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm='direction',
                                                            direction=np.deg2rad(self.sample_direction))
        self.assertAlmostEqual(mss, 0.0118, 3)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_total_snorm(self):
        """
        Check the spread-normalized total mss value for the sample arrays. 
        Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm='direction',
                                                            direction=np.deg2rad(self.sample_spread))
        self.assertAlmostEqual(mss, 0.0148, 3)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_total_fsnorm(self):
        """
        Check the f- and spread-normalized total mss value for the
        sample arrays. Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm=['frequency', 'spread'],
                                                            spread=np.deg2rad(self.sample_spread))
        self.assertAlmostEqual(mss, 0.0317, 3)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_total_fdnorm(self):
        """
        Check the f- and dir-normalized total mss value for the
        sample arrays. Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='total',
                                                            norm=['frequency', 'direction'],
                                                            direction=np.deg2rad(self.sample_direction))
        self.assertAlmostEqual(mss, 0.025, 3)
        self.assertAlmostEqual(bw, 0.47, 2)
        self.assertEqual(np.sum(freq_range_logical), 38)

    def test_mean_square_slope_dynamic(self):
        """
        Check the dynamic mss value for the sample arrays. Cross 
        validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='dynamic',
                                                            norm=None)
        self.assertAlmostEqual(mss, 0.0040, 4)
        self.assertAlmostEqual(bw, 0.117, 3)
        self.assertEqual(np.sum(freq_range_logical), 13)

    def test_mean_square_slope_dynamic_fnorm(self):
        """
        Check the f-normalized dynamic mss value for the sample arrays.  
        Cross validated with Jim Thomson's SWIFT codes.
        """
        mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
                                                            self.sample_freq,
                                                            freq_range='dynamic',
                                                            norm='frequency')
        self.assertAlmostEqual(mss, 0.034, 3)
        self.assertAlmostEqual(bw, 0.117, 3)
        self.assertEqual(np.sum(freq_range_logical), 13)

    def test_mean_square_slope_ndarray(self):
        """
        Test of ndarray functionality for mean_square_slope function. 
        Check the dynamic mss value for an ndarray made up of the sample
        arrays. Cross validated with Jim Thomson's SWIFT codes.
        """
        nd = 10
        energy_nd = np.tile(self.sample_energy, (nd,1))
        freq_nd = np.tile(self.sample_freq, (nd,1))

        mss, bw, freq_range_logical = waves.mean_square_slope(energy_nd,
                                                            freq_nd,
                                                            freq_range='dynamic',
                                                            norm='frequency')
        self.assertTrue(np.allclose(
                        mss,
                        np.repeat(0.0341, nd, axis=0),
                        rtol=1e-02,
                        atol=1e-03))
        self.assertTrue(np.allclose(
                        bw,
                        np.repeat(0.117, nd, axis=0),
                        rtol=1e-02,
                        atol=1e-03))

    def test_mean_square_slope_ndarray_multinorm(self):
        """
        Test of ndarray functionality for mean_square_slope function. 
        Check the multi-norm mss value for an ndarray made up of the 
        sample arrays. Cross validated with Jim Thomson's SWIFT codes.
        """
        nd = 10
        energy_nd = np.tile(self.sample_energy, (nd,1))
        freq_nd = np.tile(self.sample_freq, (nd,1))
        dir_nd = np.tile(self.sample_direction, (nd,1))
        spread_nd = np.tile(self.sample_spread, (nd,1))

        mss, bw, freq_range_logical = waves.mean_square_slope(energy_nd,
                                                            freq_nd,
                                                            freq_range='total',
                                                            norm=['frequency', 'spread'],
                                                            direction=np.deg2rad(dir_nd),
                                                            spread=np.deg2rad(spread_nd))
        self.assertTrue(np.allclose(
                        mss,
                        np.repeat(0.0317, nd, axis=0),
                        rtol=1e-02,
                        atol=1e-03))
        self.assertTrue(np.allclose(
                        bw,
                        np.repeat(0.47, nd, axis=0),
                        rtol=1e-02,
                        atol=1e-03))

    # energy period function tests
    def test_energy_period_value(self):
        """ 
        Check the energy period value for the sample arrays. Cross
        validated with Jim Thomson's SWIFT codes.
        """
        T_e = waves.energy_period(self.sample_energy,
                                self.sample_freq,
                                returnAsFreq=False)

        self.assertAlmostEqual(T_e,  8, 1)

    def test_energy_period_as_freq(self):
        """ 
        Check the energy frequency value for the sample arrays. Cross
        validated with Jim Thomson's SWIFT codes.
        """
        f_e = waves.energy_period(self.sample_energy,
                                self.sample_freq,
                                returnAsFreq=True)

        self.assertAlmostEqual(f_e, .125, 2)


# class TestGeneralWaveTools(unittest.TestCase): 


        # # Check the output has the same 0-dim shape as a float:
        # self.assertEqual(
        #     np.shape(knn_regression(k, data, query)), 
        #     np.shape(1.0),
        # )


        # with self.assertWarns(Warning):
        #     out = knn_regression(k, data, query)
        #     assert np.isclose(out, 30.75) 

        # self.assertAlmostEqual(
        #     knn_regression(k, data, query),
        #     15.25,
        #     2,
        # )


        # with self.assertWarns(Warning):
        #     out = knn_regression(k, data, query)
        #     assert np.isclose(out, 30.75)


        # with self.assertRaises(ValueError):
        #     knn_regression(k, data, query)

    # testing for general wave tools:

    # import matplotlib.pyplot as plt
    #TODO: testing
    # f = np.ones(1000)*0.1
    # w = 2*np.pi*f
    # h = np.linspace(0.5, 1000, len(f))
    # k = dispersion(f, h)

    # fig, ax = plt.subplots()
    # ax.plot(h, k[:, 0], color='k')
    # ax.axhline(w[0]**2 / GRAVITY)
    # ax.plot(h, w * np.sqrt(1 / (GRAVITY * h)))

    # #%%
    # f = np.linspace(0.05, 0.5, 40)

    # h = np.linspace(0.5, 10, 10)

    # f = 0.05
    # h = 0.5
    # k = dispersion(f, h)

    # k
    # #%%
    # import time
    # n = 10
    # f = np.linspace(0.05, 0.5, 40)
    # f_mat = np.tile(f, (n, 1))
    # # w = 2*np.pi*f
    # # w_mat = 2*np.pi*f_mat
    # h = np.linspace(0.5, n, len(f_mat))
    # h_mat = np.tile(h, (len(f), 1)).T

    # start = time.time()
    # wavenumber_1 = dispersion(frequency=f, depth=h, use_limits=False)
    # end = time.time()
    # print(end - start)


    # # start = time.time()
    # # wavenumber = dispersion(frequency=f_mat, depth=h, use_limits=False)
    # # end = time.time()
    # # print(end - start)

    # start = time.time()
    # wavenumber_2 = dispersion(frequency=f_mat, depth=h_mat, use_limits=True)
    # end = time.time()
    # print(end - start)

    # start = time.time()
    # wavenumber_3 = dispersion(frequency=f_mat, depth=h_mat, use_limits=False)
    # end = time.time()
    # print(end - start)

    # print(np.mean((wavenumber_2 - wavenumber_3)**2))
    # print(wavenumber_1)
    # print(wavenumber_2)
    # print(wavenumber_3)
    # #%%

    # fig, ax = plt.subplots()
    # ax.plot(h, wavenumber_1[:, 0])
    # ax.plot(h, wavenumber_2[:, 0])


if __name__ == "__main__":
     unittest.main()
