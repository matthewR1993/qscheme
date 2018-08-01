from ..utils import *
import numpy as np
import pytest


def test_get_state_norm():
    assert get_state_norm(np.array([1])) == 1
    assert get_state_norm(np.array([1, 1, 1])) == 4
    assert get_state_norm(np.array([1, 1, 1, 1])) == 10


def test_diagonal_factorials():
    pass


def test_get_state_norm_2ch():
    pass
