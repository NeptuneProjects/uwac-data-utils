# -*- coding: utf-8 -*-

from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
import locale
import logging
import math
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import polars as pl
import scipy
import scipy.signal as signal

from datautils.catalogue import RecordCatalogue
from datautils.formats.formats import FileFormat, validate_file_format
from datautils.formats.shru import read_data as read_shru_data
from datautils.formats.shru import condition_data as condition_shru_data
from datautils.query import CatalogueQuery
from datautils.signal import get_filter
from datautils.time import (
    TIME_CONVERSION_FACTOR,
    TIME_PRECISION,
    datetime_linspace,
)
from datautils.util import create_empty_data_chunk, round_away

locale.setlocale(locale.LC_ALL, "")

MAX_BUFFER = int(2e9)


class BufferExceededWarning(Warning):
    pass


class DataDiscontinuityWarning(Warning):
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
class DataStreamStats:
    channels: Optional[Union[int, list[int]]] = None
    time_init: Optional[Union[float, np.datetime64]] = None
    time_end: Optional[Union[float, np.datetime64]] = None
    sampling_rate: Optional[float] = None
    units: Optional[str] = None


@dataclass
class DataStream:
    # TODO: Enable time vector construction
    """Contains acoustic data and data statistics.

    Args:
        stats (DataStreamStats): Data statistics.
        waveform (Optional[np.ndarray], optional): Acoustic data. Defaults to None.

    Returns:
        DataStream: Data stream object.

    Raises:
        NoDataWarning: If no data is found in the object.
    """

    stats: Optional[DataStreamStats]
    waveform: Optional[np.ndarray] = None

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """Returns data and time vector sliced by time index."""
        return self.waveform[index]

    def __post_init__(self):
        """Initializes data and time vector."""
        # Set time_init to 0 if not provided
        if np.isnat(self.stats.time_init):
            self.stats.time_init = np.timedelta64(0, "us")

        # Compute sampling rate if time_init and time_end are provided
        if (
            self.stats.time_init is not None
            and self.stats.time_end is not None
            and self.stats.sampling_rate is None
        ):
            self.stats.sampling_rate = (
                self.stats.time_end - self.stats.time_init
            ) / self.num_samples

        # Set time_end if time_init and sampling rate are provided
        if (
            self.stats.time_init is not None
            and np.isnat(self.stats.time_end)
            and self.stats.sampling_rate is not None
        ):
            self.stats.time_end = self.stats.time_init + np.timedelta64(
                int(
                    TIME_CONVERSION_FACTOR * self.num_samples / self.stats.sampling_rate
                ),
                TIME_PRECISION,
            )

    def __len__(self) -> int:
        """Returns length of data."""
        return self.num_samples

    def __repr__(self) -> str:
        """Returns string representation of the object."""
        return (
            f"DataStream(waveform={self.waveform}, "
            f"channels={self.stats.channels}, "
            f"num_channels={self.num_channels}, "
            f"num_samples={self.num_samples}, "
            f"time_init={self.stats.time_init}, "
            f"time_end={self.stats.time_end}, "
            f"sampling_rate={self.stats.sampling_rate}), "
            f"units={self.stats.units}"
        )

    @property
    def num_channels(self) -> int:
        """Returns number of channels in data.

        Returns:
            int: Number of channels.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.waveform is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        return self.waveform.shape[1]

    @property
    def num_samples(self) -> int:
        """Returns number of samples in data.

        Returns:
            int: Number of samples.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.waveform is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        return self.waveform.shape[0]

    @property
    def time_vector(self) -> np.ndarray:
        """Returns time vector.

        Returns:
            np.ndarray: Time vector.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.waveform is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        return datetime_linspace(
            start=self.stats.time_init, end=self.stats.time_end, num=self.num_samples
        )

    def copy(self) -> DataStream:
        return deepcopy(self)

    def slice(
        self,
        starttime: Optional[Union[int, float, np.datetime64]] = None,
        endtime: Optional[Union[int, float, np.datetime64]] = None,
        nearest_sample: bool = True,
    ) -> DataStream:
        """Slices data by time."""
        ds = copy(self)
        ds.stats = deepcopy(self.stats)
        return ds.trim(
            starttime=starttime, endtime=endtime, nearest_sample=nearest_sample
        )

    def write(self, path: Path) -> None:
        """Writes data to file."""
        np.savez(path, waveform=self.waveform, stats=self.stats)

    def write_wav(self, path: Path) -> None:
        """Writes data to WAV file."""
        scipy.io.wavfile.write(
            path, int(self.stats.sampling_rate), self.waveform.astype(np.int32)
        )

    def decimate(
        self,
        factor: int,
        n: Optional[int] = None,
        ftype: str = "iir",
        axis: int = 0,
        zero_phase: bool = True,
    ) -> DataStream:
        """Decimates data.

        Args:
            factor (int): Decimation factor.
            n (int, optional): The order of the filter. Defaults to None.
            ftype (str, optional): The type of the filter. Defaults to 'iir'.
            axis (int, optional): The axis along which to decimate. Defaults to 0.
            zero_phase (bool, optional): Prevent phase shift by filtering forward 
                and backward. Defaults to True.

        Returns:
            DataStream: Decimated data stream.
        """

        self.waveform = signal.decimate(
            self.waveform, factor, n, ftype, axis, zero_phase
        )
        self.stats.sampling_rate = self.stats.sampling_rate / float(factor)
        return self

    def filter(self, filt_type: str, **kwargs) -> DataStream:
        """Filters data.

        Args:
            filt_type (str): Filter type.
            **kwargs: Additional keyword arguments. Frequency parameters are
                expected for bandpass, bandstop, highpass, and lowpass filters.

        Returns:
            DataStream: Filtered data stream.
        """
        func = get_filter(filt_type)
        self.waveform = func(data=self.waveform, fs=self.stats.sampling_rate, **kwargs)
        return self

    def max(self) -> np.ndarray:
        """Returns maximum value of data.

        Returns:
            np.ndarray: Maximum value of data.
        """
        _max = np.atleast_1d(self.waveform.max(axis=0))
        _min = np.atleast_1d(self.waveform.min(axis=0))

        for i in range(self.num_channels):
            if abs(_max[i]) < abs(_min[i]):
                _max[i] = _min[i]

        return _max.squeeze()

    def trim(
        self,
        starttime: Optional[Union[int, float, np.datetime64]] = None,
        endtime: Optional[Union[int, float, np.datetime64]] = None,
        pad: bool = False,
        nearest_sample: bool = True,
        fill_value=None,
    ) -> DataStream:
        """Trims data by time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            starttime (Union[int, float, np.datetime64]): Start time.
            endtime (Union[int, float, np.datetime64]): End time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            ValueError: If starttime is greater than endtime.
        """
        if starttime is not None and endtime is not None and starttime > endtime:
            raise ValueError("starttime must be less than endtime.")
        if starttime:
            self._ltrim(starttime, pad, nearest_sample, fill_value)
        if endtime:
            self._rtrim(endtime, pad, nearest_sample, fill_value)
        return self

    def _ltrim(
        self,
        starttime: Union[int, float, np.datetime64],
        pad=False,
        nearest_sample=True,
        fill_value=None,
    ) -> DataStream:
        """Trims all data of this object's waveform to given start time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            starttime (Union[int, float, np.datetime64]): Start time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            TypeError: If starttime is not of type float, int, or np.datetime64.
            Exception: If time offset between starttime and time_init is too large.
        """
        dtype = self.waveform.dtype

        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = self.stats.time_init + np.timedelta64(
                int(starttime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )
        elif not isinstance(starttime, np.datetime64):
            raise TypeError("starttime must be of type float, int, or np.datetime64.")

        if nearest_sample:
            delta = round_away(
                (starttime - self.stats.time_init)
                / np.timedelta64(1, "s")
                * self.stats.sampling_rate
            )
            if delta < 0 and pad:
                npts = abs(delta) + 10
                newstarttime = self.stats.time_init - np.timedelta64(
                    int(npts / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                    TIME_PRECISION,
                )
                newdelta = round_away(
                    (starttime - newstarttime)
                    / np.timedelta64(1, "s")
                    * self.stats.sampling_rate
                )
                delta = newdelta - npts
        else:
            delta = (
                int(
                    math.floor(
                        round(
                            (self.stats.time_init - starttime)
                            / np.timedelta64(1, "s")
                            * self.stats.sampling_rate,
                            7,
                        )
                    )
                )
                * -1
            )

        if delta > 0 or pad:
            self.stats.time_init += np.timedelta64(
                int(delta / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                TIME_PRECISION,
            )
        if delta == 0 or (delta < 0 and not pad):
            return self
        if delta < 0 and pad:
            try:
                gap = create_empty_data_chunk(
                    abs(delta), self.waveform.dtype, fill_value
                )
            except ValueError:
                raise Exception(
                    "Time offset between starttime and time_init too large."
                )
            self.waveform = np.ma.concatenate([gap, self.waveform], axis=0)
            return self
        if starttime > self.stats.time_end:
            self.waveform = np.empty(0, dtype=dtype)
            return self
        if delta > 0:
            try:
                self.waveform = self.waveform[delta:]
            except IndexError:
                self.waveform = np.empty(0, dtype=dtype)
        return self

    def _rtrim(
        self,
        endtime: Union[int, float, np.datetime64],
        pad=False,
        nearest_sample=True,
        fill_value=None,
    ) -> DataStream:
        """Trims all data of this object's waveform to given end time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            endtime (Union[int, float, np.datetime64]): End time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            TypeError: If endtime is not of type float, int, or np.datetime64.
            Exception: If time offset between endtime and time_start is too large.
        """
        dtype = self.waveform.dtype

        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = self.stats.time_end - np.timedelta64(
                int(endtime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )
        elif not isinstance(endtime, np.datetime64):
            raise TypeError("endtime must be of type float, int, or np.datetime64.")

        if nearest_sample:
            delta = round_away(
                (endtime - self.stats.time_init)
                / np.timedelta64(1, "s")
                * self.stats.sampling_rate
                - self.num_samples
                + 1
            )
        else:
            delta = (
                int(
                    math.floor(
                        round(
                            (endtime - self.stats.time_end)
                            / np.timedelta64(1, "s")
                            * self.stats.sampling_rate,
                            7,
                        )
                    )
                )
                * -1
            )

        if delta == 0 or (delta > 0 and not pad):
            return self
        if delta > 0 and pad:
            try:
                gap = create_empty_data_chunk(delta, self.waveform.dtype, fill_value)
            except ValueError:
                raise Exception(
                    "Time offset between starttime and time_start too large."
                )
            self.waveform = np.ma.concatenate([self.waveform, gap], axis=0)
            return self
        if endtime < self.stats.time_init:
            self.stats.time_init = self.stats.time_end + np.timedelta64(
                int(delta / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                TIME_PRECISION,
            )
            self.waveform = np.empty(0, dtype=dtype)
            return self
        delta = abs(delta)
        total = len(self.waveform) - delta
        if endtime == self.stats.time_init:
            total = 1
        self.waveform = self.waveform[:total]
        self.stats.time_end = self.stats.time_init + np.timedelta64(
            int(TIME_CONVERSION_FACTOR * self.num_samples / self.stats.sampling_rate),
            TIME_PRECISION,
        )
        return self


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

    def _get_conditioner(file_format: FileFormat) -> callable:
        if file_format == FileFormat.SHRU:
            return condition_shru_data
        else:
            raise ValueError(f"File format {file_format} not supported.")

    def _get_reader(file_format: FileFormat) -> callable:
        if file_format == FileFormat.SHRU:
            return read_shru_data
        else:
            raise ValueError(f"File format {file_format} not supported.")

    catalogue = RecordCatalogue().load(query.catalogue)

    num_channels = len(query.channels)
    df = select_records_by_time(catalogue.df, query.time_start, query.time_end)
    if len(df) == 0:
        raise NoDataError("No data found for the given query parameters.")
    logging.debug(f"Reading {len(df)} records.")

    filenames = [
        Path(f) for f in sorted(df.unique(subset=["filename"])["filename"].to_list())
    ]
    timestamps = sorted(df.unique(subset=["filename"])["timestamp"].to_numpy())
    fixed_gains = df.unique(subset=["filename"])["fixed_gain"].to_list()
    sensitivities = df.unique(subset=["filename"])["hydrophone_sensitivity"].to_list()
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
    expected_buffer = compute_expected_buffer(df)
    if expected_buffer > max_buffer:
        warnings.warn(
            f"Buffer length {max_buffer} is less than expected samples {expected_buffer}.",
            BufferExceededWarning,
        )

    logging.debug(f"Initializing buffer...")
    waveform = -2009.0 * np.ones((expected_buffer, num_channels))
    logging.debug(f"Buffer initialized.")

    marker = 0
    time_init = None
    time_end = None
    for filename, timestamp, fixed_gain, sensitivity in zip(
        filenames, timestamps, fixed_gains, sensitivities
    ):
        logging.debug(f"Reading {filename} at {timestamp}.")
        # Define 'time_init' for waveform:
        if time_init is None:
            time_init = timestamp
        # Check time gap between files and stop if files are not continuous:
        if (time_end is not None) and (
            abs(timestamp - time_end) / np.timedelta64(1, "s") > 1 / sampling_rate
        ):
            warnings.warn(
                "Files are not continuous; time gap between files is greater than 1/sampling_rate.\n"
                f"    Stopping at {filename} at {time_end}.",
                DataDiscontinuityWarning,
            )
            break

        # Get record numbers for the file:
        rec_ind = df.filter(pl.col("filename") == str(filename))[
            "record_number"
        ].to_list()
        logging.debug(f"Reading records {rec_ind} from {filename}.")

        # Read data from file; header is not used here:
        raw_data, _ = _get_reader(file_format=validate_file_format(filename.suffix))(
            filename=filename,
            records=rec_ind,
            channels=query.channels,
            fixed_gain=fixed_gain,
        )

        # Condition data and get units:
        data, units = _get_conditioner(
            file_format=validate_file_format(filename.suffix)
        )(raw_data, fixed_gain, sensitivity)

        # Store data in waveform & advance marker by data length:
        waveform[marker : marker + data.shape[0]] = data
        marker += data.shape[0]

        # Compute time of last point in waveform:
        time_end = timestamp + np.timedelta64(
            int(TIME_CONVERSION_FACTOR * data.shape[0] / sampling_rate), TIME_PRECISION
        )

    if marker < expected_buffer:
        waveform = waveform[:marker]

    ds = DataStream(
        stats=DataStreamStats(
            channels=query.channels,
            time_init=time_init,
            time_end=time_end,
            sampling_rate=sampling_rate,
            units=units,
        ),
        waveform=waveform,
    )
    return ds.trim(starttime=query.time_start, endtime=query.time_end)


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

    last_row_before_start = df.filter(pl.col("timestamp") < time_start)["row_nr"].max()
    row_numbers = df.filter(
        (pl.col("timestamp") >= time_start) & (pl.col("timestamp") <= time_end)
    )["row_nr"].to_list()
    row_numbers.insert(0, last_row_before_start)
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


def compute_expected_buffer(df: pl.DataFrame) -> int:
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
