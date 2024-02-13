# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

TIME_PRECISION = "us"
TIME_CONVERSION_FACTOR = 1e6


@dataclass
class ClockParameters:
    time_check_0: Optional[np.datetime64] = np.datetime64("NaT")
    time_check_1: Optional[np.datetime64] = np.datetime64("NaT")
    offset_0: float = 0.0
    offset_1: float = 0.0

    @property
    def drift_rate(self) -> float:
        """Computes clock drift rate in s/day.

        Returns:
            float: Clock drift rate [s/day].
        """
        if np.isnat(self.time_check_0) or np.isnat(self.time_check_1):
            return 0.0
        days_diff = (self.time_check_1 - self.time_check_0) / np.timedelta64(1, "D")
        offset_diff = self.offset_1 - self.offset_0
        return offset_diff / days_diff if days_diff != 0 else 0.0


def convert_timestamp_to_yyd(timestamp: np.datetime64) -> tuple[int, float]:
    """Converts a timestamp to year and year-day fraction.

    Args:
        timestamp (np.datetime64): Timestamp to convert.

    Returns:
        tuple[int, float]: Year and year-day fraction.
    """
    Y, rmndr = [timestamp.astype(f"datetime64[{t}]") for t in ["Y", TIME_PRECISION]]
    year = Y.astype(int) + 1970
    yd = (rmndr - np.datetime64(f"{year - 1}-12-31")) / np.timedelta64(1, "D")
    return year, yd


def convert_yydfrac_to_timestamp(year: int, yd: float) -> np.datetime64:
    """Converts year and year-day fraction to a timestamp.

    Args:
        year (int): Year.
        yd (float): Year-day fraction.

    Returns:
        np.datetime64: Timestamp.
    """
    base_date = np.datetime64(f"{int(year)}-01-01")
    day = int(yd)
    fraction = yd - day
    rmndr = int(fraction * 24 * 60 * 60 * TIME_CONVERSION_FACTOR)
    return base_date + np.timedelta64(day, "D") + np.timedelta64(rmndr, "us")


def convert_to_datetime(
    year: int, yd: int, minute: int, millisec: int, microsec: int
) -> np.datetime64:
    """Converts year, year-day, minute, millisecond, and microsecond to a timestamp.

    Args:
        year (int): Year.
        yd (int): Year-day.
        minute (int): Minute.
        millisec (int): Millisecond.
        microsec (int): Microsecond.

    Returns:
        np.datetime64: Timestamp.
    """
    base_date = np.datetime64(f"{year}-01-01")
    return (
        base_date
        + np.timedelta64(yd - 1, "D")
        + np.timedelta64(minute, "m")
        + np.timedelta64(millisec, "ms")
        + np.timedelta64(microsec, "us")
    )


def correct_clock_drift(
    timestamp: np.datetime64, clock: ClockParameters
) -> np.datetime64:
    """Correct clock drift in a timestamp.

    Args:
        timestamp (np.datetime64): Timestamp to correct.
        clock (ClockParameters): Clock parameters.

    Returns:
        np.datetime64: Corrected timestamp.
    """
    print(clock.time_check_0, clock.time_check_1)
    if np.isnat(clock.time_check_0) or np.isnat(clock.time_check_1):
        return timestamp

    days_diff = (timestamp - clock.time_check_0) / np.timedelta64(1, "D")
    drift = clock.drift_rate * days_diff
    return timestamp + np.timedelta64(
        int(TIME_CONVERSION_FACTOR * drift), TIME_PRECISION
    )


def convert_datetime64_to_pldatetime(np_datetime64: np.datetime64) -> str:
    np_datetime64_us = np_datetime64.astype(f"datetime64[{TIME_PRECISION}]")
    int64_us = np.int64(np_datetime64_us.view("int64"))
    return pl.lit(int64_us).cast(pl.Datetime(TIME_PRECISION))


def convert_pldatetime_to_datetime64(pl_datetime: pl.Series) -> list[np.datetime64]:
    int64_us = pl_datetime.cast(pl.Int64)
    return np.datetime64(int64_us, TIME_PRECISION)


def datetime_linspace(start: np.datetime64, end: np.datetime64, num: int) -> np.ndarray:
    start_int = start.astype("int64")
    end_int = end.astype("int64")
    return np.linspace(start_int, end_int, num, dtype="int64").astype(f"datetime64[{TIME_PRECISION}]")

# def convert_datetime64_to_pldatetime(np_datetime64: np.datetime64) -> str:
#     np_datetime64_us = np_datetime64.astype("datetime64[us]")
#     print(type(np_datetime64_us))
#     int64_us = np_datetime64_us.astype("int64")
#     print(type(int64_us))

#     # Ensure `values` is a list or an array
#     sr = pl.Series(values=[int64_us], dtype=pl.Int64)

#     return sr.cast(pl.Datetime("us"))


# def convert_datetime64_to_pldatetime(np_datetime64: np.datetime64) -> str:
#     np_datetime64_us = np_datetime64.astype("datetime64[us]")
#     print(type(np_datetime64_us))
#     int64_us = np_datetime64_us.astype("int64")
#     print(type(int64_us))
#     sr = pl.Series(values=int64_us, dtype=pl.Int64)
#     return sr.cast(pl.Datetime("us"))
