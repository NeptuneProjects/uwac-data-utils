# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Optional

import numpy as np

from datautils.hydrophone import HydrophoneSpecs
from datautils.time import ClockParameters


@dataclass
class DataSelection:
    directory: str
    glob_pattern: str
    destination: str = "."
    file_format: Optional[str] = None


@dataclass
class FileInfoQuery:
    serial: str
    data: DataSelection
    clock: ClockParameters
    hydrophones: HydrophoneSpecs


def load_query(filename: Path) -> list[FileInfoQuery]:
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
        hydrophone = HydrophoneSpecs(**params["hydrophone"])

        queries.append(FileInfoQuery(serial, data, clock, hydrophone))

    return queries
