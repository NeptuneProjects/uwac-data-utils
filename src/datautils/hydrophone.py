# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hydrophone:
    serial_number: int = 0
    fixed_gain: float = 1.0
    sensitivity: float = 1.0


@dataclass
class HydrophoneSpecs:
    fixed_gain: Optional[list[float]] = 1.0
    sensitivity: Optional[list[float]] = 1.0
    serial_number: Optional[list[int]] = 0
