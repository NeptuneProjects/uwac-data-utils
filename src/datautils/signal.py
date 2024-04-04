# -*- coding: utf-8 -*-

import warnings

import numpy as np

try:
    import cupyx.scipy.signal as signal
except ImportError:
    import scipy.signal as signal

def get_filter(filt_type: str) -> callable:
    FILTER_REGISTRY = {
        "bandpass": bandpass,
        "bandstop": bandstop,
        "highpass": highpass,
        "lowpass": lowpass,
    }
    return FILTER_REGISTRY[filt_type]


def bandpass(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    if high - 1.0 > -1e-6:
        warnings.warn(
            (
                f"Selected high corner frequency ({freqmax}) of bandpass is at or "
                f"above Nyquist ({fe}). Applying a high-pass instead."
            )
        )
        return highpass(data, freq=freqmin, fs=fs, corners=corners, zerophase=zerophase)
    if low > 1:
        raise ValueError(
            f"Selected low corner frequency ({freqmin}) of bandpass is above Nyquist ({fe})."
        )

    z, p, k = signal.iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data)
        return signal.sosfilt(sos, firstpass[::-1])[::-1]
    return signal.sosfilt(sos, data)


def bandstop(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    if high > 1:
        high = 1.0
        warnings.warn(
            (
                f"Selected high corner frequency ({freqmax}) is above "
                f"Nyquist ({fe}). Setting Nyquist as high corner."
            )
        )
    if low > 1:
        raise ValueError(
            f"Selected low corner frequency ({freqmin}) is above Nyquist ({fe})."
        )

    z, p, k = signal.iirfilter(
        corners, [low, high], btype="bandstop", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data)
        return signal.sosfilt(sos, firstpass[::-1])[::-1]
    return signal.sosfilt(sos, data)


def highpass(
    data: np.ndarray,
    freq: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    f = freq / fe
    if f > 1:
        raise ValueError(f"Selected corner frequency ({freq}) is above Nyquist ({fe}).")

    z, p, k = signal.iirfilter(
        corners, f, btype="highpass", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data)
        return signal.sosfilt(sos, firstpass[::-1])[::-1]
    return signal.sosfilt(sos, data)


def lowpass(
    data: np.ndarray,
    freq: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    f = freq / fe
    if f > 1:
        f = 1.0
        warnings.warn(
            (
                f"Selected high corner frequency ({freq}) is above "
                f"Nyquist ({fe}). Setting Nyquist as high corner."
            )
        )

    z, p, k = signal.iirfilter(
        corners, f, btype="lowpass", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data)
        return signal.sosfilt(sos, firstpass[::-1])[::-1]
    return signal.sosfilt(sos, data)
