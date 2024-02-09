# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib
from typing import Optional

import numpy as np

from datautils.formats.shru import read_header as read_shru_header
from datautils.formats.sio import read_header as read_sio_header
from datautils.formats.wav import read_header as read_wav_header


class FormatCheckerMixin:
    @classmethod
    def is_format(cls, extension: str) -> bool:
        normalized_extension = extension.lower()
        return any(normalized_extension == item.value.lower() for item in cls)


class SHRUFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "SHRU"
    D23 = ".D23"


class SIOFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "SIO"
    SIO = ".SIO"


class WAVFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "WAV"
    WAV = ".WAV"


class FileFormat(Enum):
    SHRU = SHRUFileFormat.FORMAT.value
    SIO = SIOFileFormat.FORMAT.value
    WAV = WAVFileFormat.FORMAT.value


@dataclass
class DataSelection:
    directory: str
    glob_pattern: str = "**/*.D23"
    destination: str = "."


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


@dataclass
class HydrophoneInfo:
    fixed_gain: list[float]
    sensitivity: list[float]
    serial_number: list[int]


@dataclass
class FileInfoQuery:
    serial: str
    data: DataSelection
    clock: ClockParameters
    hydrophones: HydrophoneInfo


def load_config(filename: Path) -> list[FileInfoQuery]:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    queries = []
    for serial, params in config.items():

        data = DataSelection(**params["data"])
        clock = ClockParameters(
            time_check_0=np.datetime64(params["clock"].get("time_check_0")),
            time_check_1=np.datetime64(params["clock"].get("time_check_1")),
            offset_0=params["clock"].get("offset_0", 0.0),
            offset_1=params["clock"].get("offset_1", 0.0),
        )
        hydrophone = HydrophoneInfo(**params["hydrophone"])

        queries.append(FileInfoQuery(serial, data, clock, hydrophone))

    return queries


def get_header_reader(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> callable:
    """Read file header."""
    if suffix is None and file_format is None:
        raise ValueError("An argument 'suffix' or 'file_format' must be provided.")

    if file_format is None:
        file_format = get_file_format(suffix)
    if file_format == FileFormat.SHRU:
        return read_shru_header
    if file_format == FileFormat.SIO:
        return read_sio_header
    if file_format == FileFormat.WAV:
        return read_wav_header

    raise ValueError(f"File format {file_format} is not recognized.")


def get_file_format(suffix: str) -> str:
    """Get file format from suffix."""
    if SHRUFileFormat.is_format(suffix):
        return FileFormat.SHRU
    if SIOFileFormat.is_format(suffix):
        return FileFormat.SIO
    if WAVFileFormat.is_format(suffix):
        return FileFormat.WAV
    raise ValueError(f"File format cannot be inferred from file extension '{suffix}'.")
