from ..state_configurations import *
import numpy as np
import pytest
from numpy.testing import assert_array_equal


def test_single_photon():
    with pytest.raises(IndexError):
        single_photon(0)
    with pytest.raises(IndexError):
        single_photon(1)
    assert_array_equal(single_photon(3), np.array([0, 1, 0]))
    assert_array_equal(single_photon(2), np.array([0, 1]))


def test_fock_state():
    pass
