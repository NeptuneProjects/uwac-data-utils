# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
import locale
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl

from datautils.catalogue import RecordCatalogue
from datautils.query import CatalogueQuery
from datautils.formats.shru import read_24bit_data

locale.setlocale(locale.LC_ALL, "")

MAX_BUFFER = int(2e9)


class BufferExceededWarning(Warning):
    pass


class NoDataError(Exception):
    pass


class NoDataWarning(Warning):
    pass


class DataFormat(Enum):
    """Data format."""

    CSV = "csv"
    MAT = "mat"
    NPY = "npy"
    NPZ = "npz"
    WAV = "wav"


@dataclass
class DataStream:
    """Contains acoustic data and time vector."""

    waveform: Optional[np.ndarray] = None
    # t: Optional[np.ndarray] = None
    channels: Optional[Union[int, list[int]]] = None
    time_init: Optional[Union[float, np.datetime64]] = None
    time_end: Optional[Union[float, np.datetime64]] = None
    sampling_rate: Optional[float] = None

    def __getitem__(self, index: Union[int, slice]) -> tuple[np.ndarray, np.ndarray]:
        """Returns data and time vector sliced by time index."""
        return self.waveform[index], self.t[index]

    def __post_init__(self):
        """Initializes data and time vector."""
        # Set time_init to 0 if not provided
        if np.isnat(self.time_init):
            self.time_init = np.timedelta64(0, "us")

        # Compute sampling rate if time_init and time_end are provided
        if (
            self.time_init is not None
            and self.time_end is not None
            and self.sampling_rate is None
        ):
            self.sampling_rate = (self.time_end - self.time_init) / self.num_samples

        # Set time_end if time_init and sampling rate are provided
        if (
            self.time_init is not None
            and np.isnat(self.time_end)
            and self.sampling_rate is not None
        ):
            self.time_end = self.time_init + np.timedelta64(
                int(1e6 * self.num_samples / self.sampling_rate), "us"
            )

    def __repr__(self) -> str:
        """Returns string representation of the object."""
        return (
            f"DataStream(waveform={self.waveform}, "
            f"channels={self.channels}, "
            f"num_channels={self.num_channels}, "
            f"num_samples={self.num_samples}, "
            f"time_init={self.time_init}, "
            f"time_end={self.time_end}, "
            f"sampling_rate={self.sampling_rate})"
        )

    # def load(self, filename: Path, exclude: Optional[str] = None) -> None:
    #     """Loads data from numpy file."""
    #     data = np.load(filename)
    #     try:
    #         if exclude is None or "X" not in exclude:
    #             self.waveform = data.get("X", None)
    #         if exclude is None or "t" not in exclude:
    #             self.t = data.get("t", None)
    #     except AttributeError:
    #         self.waveform = data

    @property
    def num_channels(self) -> int:
        """Returns number of channels in data."""
        if self.waveform is None:
            return NoDataWarning("No data in variable 'X'.")
        return self.waveform.shape[1]

    @property
    def num_samples(self) -> int:
        """Returns number of samples in data."""
        if self.waveform is None:
            return NoDataWarning("No data in variable 'X'.")
        return self.waveform.shape[0]

    def save(self, filename: Path) -> None:
        """Saves data to numpy file."""
        if self.waveform is None:
            NoDataWarning("No data in variable 'X' to save.")
        if self.t is None:
            NoDataWarning("No data in variable 't' to save.")
        np.savez(filename, X=self.waveform, t=self.t)

    # TODO: Implement these functionalities
    # def slice_by_time(
    #     self, time_init: Union[float, np.datetime64], time_end: Union[float, np.datetime64]
    # ) -> "DataStream":
    #     """Slices data by time."""
    #     if self.t is None:
    #         return NoDataWarning("No data in variable 't' to slice.")
    #     idx = np.where((self.t >= time_init) & (self.t <= time_end))
    #     return DataStream(X=self.X[idx], t=self.t[idx])

    # def slice_by_channel(self, channel: int) -> "DataStream":
    #     """Slices data by channel."""
    #     if self.X is None:
    #         return NoDataWarning("No data in variable 'X' to slice.")
    #     return DataStream(X=self.X[:, channel], t=self.t)

    # def slice_by_channels(self, channels: list[int]) -> "DataStream":
    #     """Slices data by channels."""
    #     if self.X is None:
    #         return NoDataWarning("No data in variable 'X' to slice.")
    #     return DataStream(X=self.X[:, channels], t=self.t)

    # def slice_by_index(self, index: Union[int, slice]) -> "DataStream":
    #     """Slices data by index."""
    #     if self.X is None:
    #         return NoDataWarning("No data in variable 'X' to slice.")
    #     return DataStream(X=self.X[index], t=self.t[index])


def read(query: CatalogueQuery, max_buffer: int = MAX_BUFFER) -> DataStream:
    """Reads data from catalogue using the query parameters.

    Args:
        query (CatalogueQuery): Query parameters.
        max_buffer (int): Maximum buffer length in samples.

    Returns:
        DataStream: Data stream object.

    Raises:
        NoDataError: If no data is found for the given query parameters.
        ValueError: If multiple sampling rates are found in the catalogue.
        BufferExceededWarning: If buffer length is less than expected samples.
    """
    catalogue = RecordCatalogue().load(query.catalogue)
    num_channels = len(query.channels)
    df = select_records_by_time(catalogue.df, query.time_start, query.time_end)
    with pl.Config(tbl_cols=df.width):
        print(df)

    if len(df) == 0:
        raise NoDataError("No data found for the given query parameters.")
    logging.debug(f"Reading {len(df)} records.")

    filenames = sorted(df.unique(subset=["filename"])["filename"].to_list())
    fixed_gains = df.unique(subset=["filename"])["fixed_gain"].to_list()
    sampling_rates = (
        df.unique(subset=["filename"])
        .unique(subset=["sampling_rate"])["sampling_rate"]
        .to_list()
    )
    if len(set(sampling_rates)) > 1:
        raise ValueError(
            "Multiple sampling rates found in the catalogue; unable to proceed."
        )
    sampling_rate = sampling_rates[0]

    report_buffer(max_buffer, num_channels, sampling_rate)
    expected_samples = compute_expected_samples(df)
    if expected_samples > max_buffer:
        raise BufferExceededWarning(
            f"Buffer length {max_buffer} is less than expected samples {expected_samples}."
        )

    logging.debug(f"Initializing buffer...")
    waveform = -2009.0 * np.ones((expected_samples, num_channels))
    logging.debug(f"Buffer initialized.")

    marker = 0
    for filename, fixed_gain in zip(filenames, fixed_gains):
        rec_ind = df.filter(pl.col("filename") == filename)["record_number"].to_list()
        logging.debug(f"Reading records {rec_ind} from {filename}.")

        # TODO: Create reader factory to read different file formats
        data, header = read_24bit_data(
            filename=filename,
            records=rec_ind,
            channels=query.channels,
            fixed_gain=fixed_gain,
        )
        waveform[marker : marker + data.shape[0]] = data
        marker += data.shape[0]

        # if marker >= buffer:
        #     break

        # TODO: Merge data and header into a single DataStream object
        # TODO: Enable time vector construction
        # TODO: Trim the waveform to the specified time range
        # TODO: Implement buffer to read data in chunks

    if marker < max_buffer:
        waveform = waveform[:marker]

    return DataStream(
        waveform=waveform,
        channels=query.channels,
        time_init=query.time_start,
        time_end=query.time_end,
        sampling_rate=sampling_rate,
    )


def select_records_by_time(
    df: pl.DataFrame, time_start: np.datetime64, time_end: np.datetime64
) -> pl.DataFrame:
    """Select files by time."""
    logging.debug(f"Selecting records by time: {time_start} to {time_end}.")
    if time_start > time_end:
        raise ValueError("time_start must be less than time_end.")
    if np.isnat(time_start) and np.isnat(time_end):
        return df
    if time_start is not None and np.isnat(time_end):
        row_numbers = df.filter(pl.col("timestamp") >= time_start)["row_nr"].to_list()
        row_numbers.insert(0, row_numbers[0] - 1)
        mask = pl.col("row_nr").is_in(row_numbers)
        return df.filter(mask)
    if np.isnat(time_start) and time_end is not None:
        return df.filter(pl.col("timestamp") <= time_end)

    row_numbers = df.filter(
        (pl.col("timestamp") >= time_start) & (pl.col("timestamp") <= time_end)
    )["row_nr"].to_list()
    row_numbers.insert(0, row_numbers[0] - 1)
    mask = pl.col("row_nr").is_in(row_numbers)
    return df.filter(mask)


def report_buffer(buffer: int, num_channels: int, sampling_rate: float) -> None:
    """Logs buffer size.

    Args:
        buffer (int): Buffer length in samples.
        num_channels (int): Number of channels.
        sampling_rate (float): Sampling rate.

    Returns:
        None
    """
    logging.debug(
        f"MAX BUFFER ---- length: {buffer:e} samples | "
        f"length per channel: {int(buffer / num_channels)} samples | "
        f"size: {8 * buffer / 1e6:n} MB | "
        f"duration: {buffer / sampling_rate / num_channels / 60:.3f} min with {num_channels} channels."
    )


def compute_expected_samples(df: pl.DataFrame) -> int:
    """Computes expected samples.

    Args:
        df (pl.DataFrame): Data frame.

    Returns:
        int: Expected samples.
    """
    expected_samples = 0
    for filename in sorted(df.unique(subset=["filename"])["filename"].to_list()):
        rec_ind = df.filter(pl.col("filename") == filename)["record_number"].to_list()
        expected_samples += df.filter(
            (pl.col("filename") == filename) & (pl.col("record_number").is_in(rec_ind))
        )["npts"].sum()
    logging.debug(f"Expected samples: {expected_samples}")
    return expected_samples
