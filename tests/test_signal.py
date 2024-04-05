#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from scipy import signal
from datautils.signal import bandpass, bandstop, get_filter, highpass, lowpass


class TestGetFilter(unittest.TestCase):
    def test_bandpass_filter(self):
        filter_func = get_filter("bandpass")
        self.assertTrue(callable(filter_func))
        self.assertEqual(filter_func.__name__, "bandpass")

    def test_bandstop_filter(self):
        filter_func = get_filter("bandstop")
        self.assertTrue(callable(filter_func))
        self.assertEqual(filter_func.__name__, "bandstop")

    def test_highpass_filter(self):
        filter_func = get_filter("highpass")
        self.assertTrue(callable(filter_func))
        self.assertEqual(filter_func.__name__, "highpass")

    def test_lowpass_filter(self):
        filter_func = get_filter("lowpass")
        self.assertTrue(callable(filter_func))
        self.assertEqual(filter_func.__name__, "lowpass")


class TestBandpass(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.fs = 10.0

    def test_bandpass_high_corner_above_nyquist(self):
        with self.assertWarns(Warning):
            result = bandpass(self.data, 1.0, 20.0, self.fs)
        expected_result = signal.sosfilt(
            signal.butter(4, 1.0 / (0.5 * self.fs), btype="high", output="sos"),
            self.data,
        )
        np.testing.assert_array_equal(result, expected_result)

    def test_bandpass_low_corner_above_nyquist(self):
        with self.assertRaises(ValueError):
            bandpass(self.data, 20.0, 1.0, self.fs)

    def test_bandpass_normal_case(self):
        result = bandpass(self.data, 1.0, 2.0, self.fs)
        sos = signal.iirfilter(
            4,
            [1.0 / (0.5 * self.fs), 2.0 / (0.5 * self.fs)],
            btype="band",
            ftype="butter",
            output="zpk",
        )
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        np.testing.assert_array_equal(result, expected_result)

    def test_bandpass_zerophase(self):
        result = bandpass(self.data, 1.0, 2.0, self.fs, zerophase=True)
        sos = signal.iirfilter(
            4,
            [1.0 / (0.5 * self.fs), 2.0 / (0.5 * self.fs)],
            btype="band",
            ftype="butter",
            output="zpk",
        )
        firstpass = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), firstpass[::-1])[::-1]
        np.testing.assert_array_equal(result, expected_result)


class TestBandstop(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.fs = 10.0

    def test_bandstop_high_corner_above_nyquist(self):
        with self.assertWarns(Warning):
            result = bandstop(self.data, 1.0, 20.0, self.fs)
        sos = signal.iirfilter(
            4,
            [1.0 / (0.5 * self.fs), 1.0 - 1e-6],
            btype="bandstop",
            ftype="butter",
            output="zpk",
        )
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_bandstop_low_corner_above_nyquist(self):
        with self.assertRaises(ValueError):
            bandstop(self.data, 20.0, 1.0, self.fs)

    def test_bandstop_normal_case(self):
        result = bandstop(self.data, 1.0, 2.0, self.fs)
        sos = signal.iirfilter(
            4,
            [1.0 / (0.5 * self.fs), 2.0 / (0.5 * self.fs)],
            btype="bandstop",
            ftype="butter",
            output="zpk",
        )
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_bandstop_zerophase(self):
        result = bandstop(self.data, 1.0, 2.0, self.fs, zerophase=True)
        sos = signal.iirfilter(
            4,
            [1.0 / (0.5 * self.fs), 2.0 / (0.5 * self.fs)],
            btype="bandstop",
            ftype="butter",
            output="zpk",
        )
        firstpass = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), firstpass[::-1])[::-1]
        np.testing.assert_array_almost_equal(result, expected_result)


class TestHighPass(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.fs = 100
        self.freq = 10

    def test_highpass_value_error(self):
        with self.assertRaises(ValueError):
            highpass(self.data, 100, 50)

    def test_highpass_output(self):
        result = highpass(self.data, self.freq, self.fs)
        sos = signal.iirfilter(
            4,
            self.freq / (0.5 * self.fs),
            btype="highpass",
            ftype="butter",
            output="zpk",
        )
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_highpass_zerophase(self):
        result = highpass(self.data, self.freq, self.fs, zerophase=True)
        sos = signal.iirfilter(
            4,
            self.freq / (0.5 * self.fs),
            btype="highpass",
            ftype="butter",
            output="zpk",
        )
        firstpass = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), firstpass[::-1])[::-1]
        np.testing.assert_array_almost_equal(result, expected_result)


class TestLowPass(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.fs = 100
        self.freq = 10
        self.corners = 4

    def test_lowpass_value_error(self):
        with self.assertRaises(ValueError):
            lowpass(self.data, 110, self.fs)

    def test_lowpass_output(self):
        result = lowpass(self.data, self.freq, self.fs)
        sos = signal.iirfilter(
            self.corners,
            self.freq / (0.5 * self.fs),
            btype="lowpass",
            ftype="butter",
            output="zpk",
        )
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_lowpass_zerophase(self):
        result = lowpass(self.data, self.freq, self.fs, zerophase=True)
        sos = signal.iirfilter(
            self.corners,
            self.freq / (0.5 * self.fs),
            btype="lowpass",
            ftype="butter",
            output="zpk",
        )
        firstpass = signal.sosfilt(signal.zpk2sos(*sos), self.data)
        expected_result = signal.sosfilt(signal.zpk2sos(*sos), firstpass[::-1])[::-1]
        np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
