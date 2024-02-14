# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Optional, Union

import numpy as np

from datautils.hydrophone import HydrophoneSpecs
from datautils.time import ClockParameters


@dataclass
class CatalogueQuery:
    catalogue: Path
    destination: Path
    time_start: Optional[Union[float, np.datetime64]] = None
    time_end: Optional[Union[float, np.datetime64]] = None
    channels: Optional[Union[int, list[int]]] = None

    def __repr__(self):
        return (
            f"CatalogueQuery(catalogue={self.catalogue},"
            f"destination={self.destination}, "
            f"time_start={self.time_start}, "
            f"time_end={self.time_end}, "
            f"channels={self.channels})"
        )


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
    clock: ClockParameters = field(default_factory=ClockParameters)
    hydrophones: HydrophoneSpecs = field(default_factory=HydrophoneSpecs)


def load_catalogue_query(filename: Path) -> list[CatalogueQuery]:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    queries = []
    for params in config.values():
        queries.append(
            CatalogueQuery(
                catalogue=Path(params.get("catalogue")),
                destination=params.get("destination", Path.cwd()),
                time_start=np.datetime64(params.get("time_start", None)),
                time_end=np.datetime64(params.get("time_end", None)),
                channels=params.get("channels", None),
            )
        )

    return queries


def load_file_query(filename: Path) -> list[FileInfoQuery]:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    queries = []
    for serial, params in config.items():
        data = DataSelection(**params["data"])
        clock_params = params.get("clock", {})
        clock = ClockParameters(
            time_check_0=np.datetime64(clock_params.get("time_check_0", "NaT")),
            time_check_1=np.datetime64(clock_params.get("time_check_1", "NaT")),
            offset_0=clock_params.get("offset_0", 0.0),
            offset_1=clock_params.get("offset_1", 0.0),
        )
        hydrophone = HydrophoneSpecs(**params.get("hydrophone", {}))
        queries.append(FileInfoQuery(serial, data, clock, hydrophone))

    return queries
