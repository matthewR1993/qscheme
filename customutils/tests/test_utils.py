import numpy as np
from numpy.testing import assert_array_equal

from ..utils import *


def test_get_state_norm():
    assert get_state_norm(np.array([1])) == 1
    assert get_state_norm(np.array([1, 1, 1])) == 4
    assert get_state_norm(np.array([1, 1, 1, 1])) == 10


def test_diagonal_factorials():
    assert_array_equal(diagonal_factorials(0), np.zeros((0, 0)))
    assert_array_equal(diagonal_factorials(1), np.array([[1]]))
    assert_array_equal(diagonal_factorials(5), np.array([[1, 0, 0, 0, 0],
                                                         [0, 1, 0, 0, 0],
                                                         [0, 0, sqrt(2), 0, 0],
                                                         [0, 0, 0, sqrt(6), 0],
                                                         [0, 0, 0, 0, sqrt(24)]]
                                                        ))


def test_get_state_norm_2ch():
    state1 = np.array([[1, 0], [0, 0]])
    assert get_state_norm_2ch(state1) == 1
    state2 = np.array([[1, 1], [1, 1]])
    assert get_state_norm_2ch(state2) == 4
