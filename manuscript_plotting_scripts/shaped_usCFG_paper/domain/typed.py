"""`TypedLaw`, a law between base_core quantities, shared by the domain modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from base_core.quantities.enums import Prefix


@dataclass(frozen=True)
class TypedLaw:
    """A law y(x) between base_core quantities, e.g. a beat frequency against delay.

    Called with a sequence of typed x (``Time``, ``Length``, ...) it returns a list of
    typed y, converting once at the boundary: x is read in ``x_prefix`` units and y
    is built as ``y_type(value, y_prefix)``.  With ``y_type=None`` y is a plain float
    (a signal level).  ``numpy`` is the same law on bare floats in those units; it is
    what hot loops and dense drawn curves use.
    """
    numpy: Callable[[np.ndarray], np.ndarray]
    x_prefix: Prefix
    y_type: type | None
    y_prefix: Prefix = Prefix.NONE

    def __call__(self, xs: Sequence[float]) -> list:
        y = self.numpy(np.array([x.value(self.x_prefix) for x in xs], dtype=float))
        if self.y_type is None:
            return [float(v) for v in y]
        return [self.y_type(v, self.y_prefix) for v in y]
