""" 
Unit tests for the buoy submodule.

TODO:
    - 

"""

import unittest

import numpy as np
from littlebuoybigwaves import buoy


class TestBuoy(unittest.TestCase):
    """ Unit tests for spectral wave tools submodule """

    def setUp(self):
        self.sample_energy_density = np.array([
            2.76800512e-01, 5.12624640e-01, 6.66412032e-01, 9.12480256e-01,
            3.19876096e+00, 1.61682063e+01, 3.74831473e+01, 4.17071995e+01,
            3.64374098e+01, 3.24389376e+01, 2.08741089e+01, 1.32564900e+01,
            9.10423040e+00, 6.29503898e+00, 3.69089741e+00, 3.40381082e+00,
            3.18851686e+00, 2.36831744e+00, 1.69166131e+00, 1.52762982e+00,
            1.12777421e+00, 7.89425152e-01, 7.99711232e-01, 6.35637760e-01,
            4.30587904e-01, 3.79325440e-01, 3.58837248e-01, 2.15293952e-01,
            1.84561664e-01, 1.53787392e-01, 1.33299200e-01, 1.53787392e-01,
            1.12769024e-01, 6.15065600e-02, 4.10183680e-02, 3.07742720e-02,
            2.04881920e-02, 1.02440960e-02
         ])
        self.sample_frequency = np.array([
            0.0293 , 0.03906, 0.04883, 0.05859, 0.06836, 0.07813, 0.08789,
            0.09766, 0.10742, 0.11719, 0.12695, 0.13672, 0.14648, 0.15625,
            0.16602, 0.17578, 0.18555, 0.19531, 0.20508, 0.21484, 0.22461,
            0.23438, 0.24414, 0.25391, 0.26367, 0.27344, 0.2832 , 0.29297,
            0.30273, 0.3125 , 0.32227, 0.33203, 0.35156, 0.38086, 0.41016,
            0.43945, 0.46875, 0.49805
        ])
        self.sample_direction = np.array([
            110.8418918 , 130.97432502, 146.01442946, 156.47810425,
            162.88124852, 165.73976141, 165.56954204, 162.88648954,
            158.20650305, 152.04548169, 147.22737812, 140.42742977,
            132.24748889, 124.38718195, 119.38618452, 116.43286114,
            114.06112467, 113.17634149, 111.36543768, 109.05168062,
            108.27187828, 108.21629482, 108.57091663, 109.79254242,
            109.91916072, 109.87350943, 111.42516799, 112.38128329,
            110.68930493, 110.11509256, 109.3032622 , 108.24115868,
            106.91612683, 105.3155115 , 103.42665751, 101.23690972,
            98.73361295,  95.90411204
        ])
        self.sample_drift_direction = 335.987
        self.sample_drift_speed = 0.643
        self.sample_energy_density_int = np.array([
            2.67657975e-01, 4.80423372e-01, 6.22833146e-01, 8.33767919e-01,
            2.59571778e+00, 1.20298844e+01, 2.83437619e+01, 3.67293772e+01,
            3.55040002e+01, 3.14432226e+01, 2.59108768e+01, 1.73501541e+01,
            1.16670571e+01, 8.37749935e+00, 5.92124630e+00, 3.56680471e+00,
            3.17283179e+00, 2.93920676e+00, 2.36798420e+00, 1.79046789e+00,
            1.46954218e+00, 1.21853226e+00, 8.93257534e-01, 6.73124807e-01,
            6.60332777e-01, 5.47442009e-01, 4.16299651e-01, 3.32352088e-01,
            2.95676839e-01, 2.70323438e-01, 1.96105971e-01, 1.65479641e-01,
            1.23051983e-01, 1.13749486e-01, 7.34864354e-02, 4.60120027e-02,
            3.34592631e-02, 2.67918379e-02
         ])

    # mean square slope function tests
    def test_doppler_adjustment(self):
        """
        Test Doppler adjustment. Cross-validated using source code from Colosi
        et al. (2023).
        """
        energy_density_int, frequency_int = buoy.doppler_adjust(
            energy_density_obs=self.sample_energy_density,
            frequency_obs=self.sample_frequency,
            drift_speed=self.sample_drift_speed,
            drift_direction_going=self.sample_drift_direction,
            wave_direction_coming=self.sample_direction,
            frequency_cutoff=1,
            interpolate=True,
        )
        energy_density_obs_close = np.allclose(energy_density_int,
                                               self.sample_energy_density_int,
                                               rtol=0.01,
                                               equal_nan=True)
        self.assertTrue(energy_density_obs_close)

    # def test_mean_square_slope_length_1d(self):
    #     """
    #     Check for proper output length. For 1-dimensional energy and
    #     frequency arrays, the mean square slope is a single value.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq)
    #     self.assertEqual(mss.ndim, 0)
    #     self.assertEqual(bw.ndim, 0)
    #     self.assertEqual(np.shape(freq_range_logical),
    #                      np.shape(self.sample_freq))

    # def test_mean_square_slope_type(self):
    #     """
    #     Check for proper output type.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq)
    #     self.assertIsInstance(mss, float)
    #     self.assertIsInstance(bw, float)
    #     self.assertIsInstance(freq_range_logical, np.ndarray)
    #     self.assertIsInstance(freq_range_logical[0], np.bool_)

    # def test_mean_square_slope_total(self):
    #     """
    #     Check the total mss value for the sample arrays. Cross 
    #     validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm=None)
    #     self.assertAlmostEqual(mss, 0.0121, 4)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_total_fnorm(self):
    #     """
    #     Check the f-normalized total mss value for the sample arrays. 
    #     Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm='frequency')
    #     self.assertAlmostEqual(mss, 0.0259, 3)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_total_dnorm(self):
    #     """
    #     Check the dir-normalized total mss value for the sample arrays. 
    #     Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm='direction',
    #                                                         direction=np.deg2rad(self.sample_direction))
    #     self.assertAlmostEqual(mss, 0.0118, 3)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_total_snorm(self):
    #     """
    #     Check the spread-normalized total mss value for the sample arrays. 
    #     Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm='direction',
    #                                                         direction=np.deg2rad(self.sample_spread))
    #     self.assertAlmostEqual(mss, 0.0148, 3)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_total_fsnorm(self):
    #     """
    #     Check the f- and spread-normalized total mss value for the
    #     sample arrays. Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm=['frequency', 'spread'],
    #                                                         spread=np.deg2rad(self.sample_spread))
    #     self.assertAlmostEqual(mss, 0.0317, 3)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_total_fdnorm(self):
    #     """
    #     Check the f- and dir-normalized total mss value for the
    #     sample arrays. Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='total',
    #                                                         norm=['frequency', 'direction'],
    #                                                         direction=np.deg2rad(self.sample_direction))
    #     self.assertAlmostEqual(mss, 0.025, 3)
    #     self.assertAlmostEqual(bw, 0.47, 2)
    #     self.assertEqual(np.sum(freq_range_logical), 38)

    # def test_mean_square_slope_dynamic(self):
    #     """
    #     Check the dynamic mss value for the sample arrays. Cross 
    #     validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='dynamic',
    #                                                         norm=None)
    #     self.assertAlmostEqual(mss, 0.0040, 4)
    #     self.assertAlmostEqual(bw, 0.117, 3)
    #     self.assertEqual(np.sum(freq_range_logical), 13)

    # def test_mean_square_slope_dynamic_fnorm(self):
    #     """
    #     Check the f-normalized dynamic mss value for the sample arrays.  
    #     Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     mss, bw, freq_range_logical = waves.mean_square_slope(self.sample_energy,
    #                                                         self.sample_freq,
    #                                                         freq_range='dynamic',
    #                                                         norm='frequency')
    #     self.assertAlmostEqual(mss, 0.034, 3)
    #     self.assertAlmostEqual(bw, 0.117, 3)
    #     self.assertEqual(np.sum(freq_range_logical), 13)

    # def test_mean_square_slope_ndarray(self):
    #     """
    #     Test of ndarray functionality for mean_square_slope function. 
    #     Check the dynamic mss value for an ndarray made up of the sample
    #     arrays. Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     nd = 10
    #     energy_nd = np.tile(self.sample_energy, (nd,1))
    #     freq_nd = np.tile(self.sample_freq, (nd,1))

    #     mss, bw, freq_range_logical = waves.mean_square_slope(energy_nd,
    #                                                         freq_nd,
    #                                                         freq_range='dynamic',
    #                                                         norm='frequency')
    #     self.assertTrue(np.allclose(
    #                     mss,
    #                     np.repeat(0.0341, nd, axis=0),
    #                     rtol=1e-02,
    #                     atol=1e-03))
    #     self.assertTrue(np.allclose(
    #                     bw,
    #                     np.repeat(0.117, nd, axis=0),
    #                     rtol=1e-02,
    #                     atol=1e-03))

    # def test_mean_square_slope_ndarray_multinorm(self):
    #     """
    #     Test of ndarray functionality for mean_square_slope function. 
    #     Check the multi-norm mss value for an ndarray made up of the 
    #     sample arrays. Cross validated with Jim Thomson's SWIFT codes.
    #     """
    #     nd = 10
    #     energy_nd = np.tile(self.sample_energy, (nd,1))
    #     freq_nd = np.tile(self.sample_freq, (nd,1))
    #     dir_nd = np.tile(self.sample_direction, (nd,1))
    #     spread_nd = np.tile(self.sample_spread, (nd,1))

    #     mss, bw, freq_range_logical = waves.mean_square_slope(energy_nd,
    #                                                         freq_nd,
    #                                                         freq_range='total',
    #                                                         norm=['frequency', 'spread'],
    #                                                         direction=np.deg2rad(dir_nd),
    #                                                         spread=np.deg2rad(spread_nd))
    #     self.assertTrue(np.allclose(
    #                     mss,
    #                     np.repeat(0.0317, nd, axis=0),
    #                     rtol=1e-02,
    #                     atol=1e-03))
    #     self.assertTrue(np.allclose(
    #                     bw,
    #                     np.repeat(0.47, nd, axis=0),
    #                     rtol=1e-02,
    #                     atol=1e-03))

    # # energy period function tests
    # def test_energy_period_value(self):
    #     """ 
    #     Check the energy period value for the sample arrays. Cross
    #     validated with Jim Thomson's SWIFT codes.
    #     """
    #     T_e = waves.energy_period(self.sample_energy,
    #                             self.sample_freq,
    #                             returnAsFreq=False)

    #     self.assertAlmostEqual(T_e,  8, 1)

    # def test_energy_period_as_freq(self):
    #     """ 
    #     Check the energy frequency value for the sample arrays. Cross
    #     validated with Jim Thomson's SWIFT codes.
    #     """
    #     f_e = waves.energy_period(self.sample_energy,
    #                             self.sample_freq,
    #                             returnAsFreq=True)

    #     self.assertAlmostEqual(f_e, .125, 2)


if __name__ == "__main__":
     unittest.main()
