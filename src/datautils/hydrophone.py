# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class Hydrophone:
    serial_number: int = 0
    fixed_gain: float = 1.0
    sensitivity: float = 1.0


@dataclass
class HydrophoneSpecs:
    serial_number: list[int]
    fixed_gain: list[float]
    sensitivity: list[float]
