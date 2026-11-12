"""Source (data loader) interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..member import StepFields


@dataclass
class SourceGrid:
    """lat/lon of the source grid.  Either 1-D regular vectors or 2-D arrays."""
    lat: np.ndarray
    lon: np.ndarray
    key: str

    @property
    def shape(self):
        return (self.lat.size, self.lon.size) if self.lat.ndim == 1 else self.lat.shape


class Source(ABC):
    """One model system (e.g. GEFS).  A single instance serves every member of its group, so grids
    and regridders are built once."""

    name: str = "source"
    step_hours: int = 3
    src_dx_km: float = 25.0
    native_ptype: bool = False          # model provides categorical rain/snow/sleet/fzra fields

    @abstractmethod
    def grid(self) -> SourceGrid: ...

    @abstractmethod
    def load_steps(self, init: datetime, valid_start: datetime, valid_end: datetime, member) -> list[StepFields]:
        """Native steps whose valid time is in (valid_start, valid_end].  Each StepFields.qpf_mm is
        the accumulation over that step."""
