# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hydrophone:
    """Hydrophone.

    Args:
        serial_number (int, optional): Serial number. Defaults to 0.
        fixed_gain (float, optional): Fixed gain [dB]. Defaults to 0.0.
        sensitivity (float, optional): Sensitivity [dB]. Defaults to 0.0.
    """

    serial_number: int = 0
    fixed_gain: float = 0.0
    sensitivity: float = 0.0


@dataclass
class HydrophoneSpecs:
    """Hydrophone specifications.

    Args:
        fixed_gain (Optional[list[float]], optional): Fixed gain [dB]. Defaults to 0.0.
        sensitivity (Optional[list[float]], optional): Sensitivity [dB]. Defaults to 0.0.
        serial_number (Optional[list[int]], optional): Serial number. Defaults to 0.
    """

    fixed_gain: Optional[list[float]] = 0.0
    sensitivity: Optional[list[float]] = 0.0
    serial_number: Optional[list[int]] = 0
