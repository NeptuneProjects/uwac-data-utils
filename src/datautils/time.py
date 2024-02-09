# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np


@dataclass
class ClockParameters:
    time_check_0: np.datetime64
    time_check_1: np.datetime64
    offset_0: float
    offset_1: float


def convert_timestamp_to_yyd(timestamp: np.datetime64) -> tuple[int, float]:
    Y, us = [timestamp.astype(f"datetime64[{t}]") for t in ["Y", "us"]]
    year = Y.astype(int) + 1970
    yd = (us - np.datetime64(f"{year - 1}-12-31")) / np.timedelta64(1, "D")
    return year, yd


def convert_to_datetime(
    year: int, yd: int, minute: int, millisec: int, microsec: int
) -> np.datetime64:
    base_date = np.datetime64(f"{year}-01-01")
    return (
        base_date
        + np.timedelta64(yd - 1, "D")
        + np.timedelta64(minute, "m")
        + np.timedelta64(millisec, "ms")
        + np.timedelta64(microsec, "us")
    )


# def correct_clock_drift(
#     timestamp: np.datetime64, clock: query.ClockParameters
# ) -> np.datetime64:
#     """Correct clock drift in a timestamp."""
#     days_diff = (timestamp - clock.time_check_0) / np.timedelta64(1, "D")
#     drift = clock.drift_rate * days_diff
#     return timestamp + np.timedelta64(int(1e6 * drift), "us")


# def get_timestamp(drh: DataRecordHeader) -> np.datetime64:
#     """Return the timestamp of a data record header."""
#     year = drh.date[0]
#     yd = drh.date[1]
#     minute = drh.time[0]
#     millisec = drh.time[1]
#     microsec = drh.microsec
#     return convert_to_datetime(year, yd, minute, millisec, microsec)
