# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


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
