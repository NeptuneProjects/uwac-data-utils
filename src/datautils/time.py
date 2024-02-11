# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class ClockParameters:
    time_check_0: np.datetime64
    time_check_1: np.datetime64
    offset_0: float
    offset_1: float

    @property
    def drift_rate(self) -> float:
        """Computes clock drift rate in s/day."""
        days_diff = (self.time_check_1 - self.time_check_0) / np.timedelta64(1, "D")
        offset_diff = self.offset_1 - self.offset_0
        return offset_diff / days_diff if days_diff != 0 else 0.0


def convert_timestamp_to_yyd(timestamp: np.datetime64) -> tuple[int, float]:
    Y, us = [timestamp.astype(f"datetime64[{t}]") for t in ["Y", "us"]]
    year = Y.astype(int) + 1970
    yd = (us - np.datetime64(f"{year - 1}-12-31")) / np.timedelta64(1, "D")
    return year, yd


def convert_yydfrac_to_timestamp(year: int, yd: float) -> np.datetime64:
    base_date = np.datetime64(f"{int(year)}-01-01")
    day = int(yd)
    fraction = yd - day
    microseconds = int(fraction * 24 * 60 * 60 * 1e6)
    return base_date + np.timedelta64(day, "D") + np.timedelta64(microseconds, "us")


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


def correct_clock_drift(
    timestamp: np.datetime64, clock: ClockParameters
) -> np.datetime64:
    """Correct clock drift in a timestamp."""
    days_diff = (timestamp - clock.time_check_0) / np.timedelta64(1, "D")
    drift = clock.drift_rate * days_diff
    return timestamp + np.timedelta64(int(1e6 * drift), "us")


def convert_datetime64_to_pldatetime(np_datetime64: np.datetime64) -> str:
    np_datetime64_us = np_datetime64.astype("datetime64[us]")
    int64_us = np_datetime64_us.astype("int64")
    return pl.Series(int64_us).cast(pl.Datetime("us"))

def convert_pldatetime_to_datetime64(pl_datetime: pl.Series) -> list[np.datetime64]:
    int64_us = pl_datetime.cast(pl.Int64)
    return np.datetime64(int64_us, "us")
