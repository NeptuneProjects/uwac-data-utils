# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

from datautils.catalogue import RecordCatalogue, get_timestamp, read_headers
from datautils.query import CatalogueQuery


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
        if self.time_init is None:
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
            and self.time_end is None
            and self.sampling_rate is not None
        ):
            self.time_end = self.time_init + np.timedelta64(
                int(1e6 * self.num_samples / self.sampling_rate), "us"
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


# class DataHandler(Protocol):
#     def convert(self) -> Any:
#         """Converts data to specified formats."""
#         ...

#     def load_merged(self) -> Any:
#         """Loads merged numpy file."""
#         ...

#     def merge_numpy_files(self) -> Any:
#         """Merges numpy files."""
#         ...


def read(catalogue: RecordCatalogue, query: CatalogueQuery) -> DataStream:
    print("This will be the primary reading function.")
    return


# def read_data_from_catalogue(query: CatalogueQuery) -> DataStream:
#     # TODO: Write function that takes a catalogue query and returns a DataStream object
#     """Loads data from file."""
#     # 1. Load catalogue:
#     catalogue = read_catalogue(query.catalogue)

#     # 2. Filter files by time:
#     selected_files = select_files_by_time(
#         catalogue.filenames, query.time_start, query.time_end
#     )
#     print(catalogue.filenames)
#     print(selected_files)

#     # 3. Load data from files:
#     # read(catalogue.filenames, query.time_start, query.time_end, query.channels)

#     pass



def select_files_by_time(
    filenames: list[Path], time_start: np.datetime64, time_end: np.datetime64
) -> list[Path]:
    """Select files by time."""
    if time_start > time_end:
        raise ValueError("time_start must be less than time_end.")
    if np.isnat(time_start) and np.isnat(time_end):
        return filenames
    if time_start is not None and np.isnat(time_end):
        return [
            f for f in filenames if get_timestamp(read_headers(f)[0][0]) >= time_start
        ]
    if np.isnat(time_start) and time_end is not None:
        return [
            f for f in filenames if get_timestamp(read_headers(f)[0][0]) <= time_end
        ]
    return [
        f
        for f in filenames
        if time_start <= get_timestamp(read_headers(f)[0][0]) <= time_end
    ]
