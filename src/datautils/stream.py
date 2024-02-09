# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Optional, Protocol, Union

import numpy as np


class NoDataWarning(Warning):
    pass


class DataFormat(Enum):
    """Data format."""

    CSV = "csv"
    MAT = "mat"
    NPY = "npy"
    NPZ = "npz"
    WAV = "wav"


class DataHandler(Protocol):
    def convert(self) -> Any:
        """Converts data to specified formats."""
        ...

    def load_merged(self) -> Any:
        """Loads merged numpy file."""
        ...

    def merge_numpy_files(self) -> Any:
        """Merges numpy files."""
        ...


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
        if self.time_init is None:
            self.time_init = 0

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
            and self.time_end is None
            and self.sampling_rate is not None
        ):
            self.time_end = self.time_init + self.num_samples * self.sampling_rate

    def load(
        self, filename: Union[str, bytes, os.PathLike], exclude: Optional[str] = None
    ) -> None:
        """Loads data from numpy file."""
        data = np.load(filename)
        try:
            if exclude is None or "X" not in exclude:
                self.waveform = data.get("X", None)
            if exclude is None or "t" not in exclude:
                self.t = data.get("t", None)
        except AttributeError:
            self.waveform = data

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

    def save(self, filename: Union[str, bytes, os.PathLike]) -> None:
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
