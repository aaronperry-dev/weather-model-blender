"""Per-bin aggregates (potential snowfall, snow level) that feed the browser's terrain adjustment."""
import numpy as np
import pytest

from pwpf_blender.constants import SNOW_LEVEL_WARM
from pwpf_blender.pipeline import AuxAccumulator

SHAPE = (3, 4)


def _aux(pot, q, sl):
    return {"pot": np.full(SHAPE, pot, np.float32), "q": np.full(SHAPE, q, np.float32), "sl": np.full(SHAPE, sl, np.float32)}


def test_weighted_mean_potential_and_qpf_weighted_snow_level():
    acc = AuxAccumulator(4, SHAPE)
    acc.add(0, 1.0, _aux(2.0, 1.0, 500.0))
    acc.add(0, 3.0, _aux(6.0, 3.0, 900.0))            # weight 3 member counts three times
    acc.add(1, 1.0, _aux(1.0, 2.0, 700.0))
    pot, sl = acc.window(0, 2)
    assert pot.shape == SHAPE and pot.dtype == np.float32
    assert pot[0, 0] == pytest.approx((2 * 1 + 6 * 3) / 4 + 1.0)                  # per-bin weighted mean, summed over bins
    assert sl[0, 0] == pytest.approx((500 * 1 + 900 * 9 + 700 * 2) / (1 + 9 + 2), rel=1e-5)   # QPF-weighted (w * q)


def test_members_without_signal_dilute_potential_but_not_the_snow_level():
    acc = AuxAccumulator(2, SHAPE)
    acc.add(0, 1.0, _aux(4.0, 2.0, 800.0))
    acc.add(0, 1.0, None)                              # delivered the bin but produced nothing
    pot, sl = acc.window(0, 1)
    assert pot[0, 0] == pytest.approx(2.0)
    assert sl[0, 0] == pytest.approx(800.0)


def test_empty_window_gets_the_warm_sentinel():
    acc = AuxAccumulator(3, SHAPE)
    acc.add(1, 1.0, _aux(1.0, 1.0, 300.0))
    pot, sl = acc.window(2, 3)
    assert (pot == 0).all() and (sl == SNOW_LEVEL_WARM).all()
