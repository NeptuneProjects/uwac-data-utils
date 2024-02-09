# -*- coding: utf-8 -*-

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

import numpy as np
import scipy

from datautils.time import convert_timestamp_to_yyd


@dataclass
class Catalogue:
    filenames: list[Path]
    timestamps: list[list[np.datetime64]]
    timestamps_orig: list[list[np.datetime64]]
    rhfs_orig: float
    rhfs: float
    fixed_gain: list[float]
    hydrophone_sensitivity: list[float]
    hydrophone_SN: list[str]

    def save_to_json(self, savepath: Path):
        mdict = {
            "SHRU": {
                "filenames": [str(f) for f in self.filenames],
                "timestamps": [t.tolist() for t in self.timestamps],
                "timestamps_orig": [t.tolist() for t in self.timestamps_orig],
                "rhfs_orig": self.rhfs_orig,
                "rhfs": self.rhfs,
                "fixed_gain": self.fixed_gain,
                "hydrophone_sensitivity": self.hydrophone_sensitivity,
                "hydrophone_SN": self.hydrophone_SN,
            }
        }
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        with open(savepath, "w") as f:
            json.dump(mdict, f, indent=4)

    def save_to_mat(self, savepath: Path):
        mdict = {
            "SHRU": {
                "filenames": [str(f) for f in self.filenames],
                "timestamps": self._to_ydarray(self.timestamps),
                "timestamps_orig": self._to_ydarray(self.timestamps_orig),
                "rhfs_orig": self.rhfs_orig,
                "rhfs": self.rhfs,
                "fixed_gain": self.fixed_gain,
                "hydrophone_sensitivity": self.hydrophone_sensitivity,
                "hydrophone_SN": self.hydrophone_SN,
            }
        }
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(savepath, mdict)

    def _to_ydarray(self, list_of_datetimes: list[list[np.datetime64]]) -> np.ndarray:
        # 2 x M x N
        # L = number of datetime elements
        # M = number of records
        # N = number of files
        L = 2
        M = max(len(dt) for dt in list_of_datetimes)
        N = len(list_of_datetimes)
        arr = np.zeros((L, M, N), dtype=np.float64)

        for i, dt in enumerate(list_of_datetimes):
            for j, d in enumerate(dt):
                year, yd_decimal = convert_timestamp_to_yyd(d)
                arr[0, j, i] = year
                arr[1, j, i] = yd_decimal

        return arr


# def build_fileinfo(fquery: list[query.FileInfoQuery]) -> None:
#     for fq in fquery:
#         fi = read_shru(fq)
#         fi.save_to_mat(savepath=Path(fq.data.destination) / f"{fq.serial}_FileInfo.mat")
