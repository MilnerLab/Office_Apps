# domain/config.py
from dataclasses import dataclass

from base_core.math.models import Angle, Point, Range


@dataclass
class AnalysisConfig:
    center: Point
    angle: Angle
    analysis_zone: Range
    transform_parameter: float