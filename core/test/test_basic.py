import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from ..state_configurations import single_photon
from ..basic import *


def test_bs2x2_transform():
    series_length = 2
    state1 = np.tensordot(single_photon(series_length), single_photon(series_length), axes=0)
    t1, r1 = 1, 0
    assert_array_equal(bs2x2_transform(t1, r1, state1), np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    t2, r2 = 0, 1
    assert_array_equal(bs2x2_transform(t2, r2, state1), np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]]))
    t3, r3 = sqrt(0.5), sqrt(0.5)
    assert_allclose(bs2x2_transform(t3, r3, state1), np.array([[0, 0, 0.5j], [0, 0, 0], [0.5j, 0, 0]]))
    t4, r4 = sqrt(0.73), sqrt(0.27)
    assert_allclose(bs2x2_transform(t4, r4, state1), np.array([[0, 0, 1j*t4*r4], [0, t4**2 - r4**2, 0], [1j*t4*r4, 0, 0]]))


def test_two_bs2x4_transform():
    series_length = 2
    state1 = np.tensordot(single_photon(series_length), single_photon(series_length), axes=0)
    t1, r1 = 0, 1
    t2, r2 = 0, 1
    two_bs2x4_transform(t1, r1, t2, r2, state1)
    pass


def test_detection():
    pass
