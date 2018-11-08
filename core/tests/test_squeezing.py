import cmath

from ..squeezing import *


def test_coord_aver():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert coord_aver(rho1, channel=1) == 0
    assert coord_aver(rho1, channel=2) == 0

    t = 0.379
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2

    assert coord_aver(rho2, channel=1) == 0
    assert coord_aver(rho2, channel=2) == 0

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3

    assert coord_aver(rho3, channel=1) == 0
    assert coord_aver(rho3, channel=2) == 0

    rho4 = np.zeros((size,) * 4, dtype=complex)
    rho4[0, 0, 1, 0] = 2
    rho4[1, 0, 0, 0] = 3 + 5j
    rho4[2, 1, 1, 1] = 1 - 2j
    rho4[0, 0, 2, 1] = 1 + 1j
    rho4[1, 0, 1, 1] = 8
    rho4[1, 1, 1, 0] = 7 - 5j
    rho4[1, 1, 1, 2] = 11 + 5j

    assert coord_aver(rho4, channel=1) == 0.5 * (rho4[0, 0, 1, 0] + rho4[1, 0, 0, 0] + rho4[2, 1, 1, 1] * sqrt(2))
    assert coord_aver(rho4, channel=2) == 0.5 * (rho4[1, 0, 1, 1] + rho4[1, 1, 1, 0] + rho4[1, 1, 1, 2] * sqrt(2))


def test_impulse_aver():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert impulse_aver(rho1, channel=1) == 0
    assert impulse_aver(rho1, channel=2) == 0

    t = 0.761
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2

    assert impulse_aver(rho2, channel=1) == 0
    assert impulse_aver(rho2, channel=2) == 0

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3

    assert impulse_aver(rho3, channel=1) == 0
    assert impulse_aver(rho3, channel=2) == 0

    rho4 = np.zeros((size,) * 4, dtype=complex)
    rho4[0, 0, 1, 0] = 2
    rho4[1, 0, 0, 0] = 3 + 5j
    rho4[2, 1, 1, 1] = 1 - 2j
    rho4[0, 0, 2, 1] = 1 + 1j
    rho4[1, 0, 1, 1] = 8
    rho4[1, 1, 1, 0] = 7 - 5j
    rho4[1, 1, 1, 2] = 11 + 5j

    assert impulse_aver(rho4, channel=1) == - 0.5j * (- rho4[0, 0, 1, 0] + rho4[1, 0, 0, 0] + rho4[2, 1, 1, 1] * sqrt(2))
    assert impulse_aver(rho4, channel=2) == - 0.5j * (- rho4[1, 0, 1, 1] + rho4[1, 1, 1, 0] - rho4[1, 1, 1, 2] * sqrt(2))


def test_prod_coord_aver():
    size = 4

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert prod_coord_aver(rho1) == 0

    t = 0.761
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[1, 0, 1, 0] = 2 * t**2
    rho2[1, 0, 0, 1] = -2j * t * r
    rho2[0, 1, 1, 0] = 2j * t * r
    rho2[0, 1, 0, 1] = 2 * r ** 2
    rho2[0, 0, 0, 0] = 1
    rho2 = rho2 / 3
    assert prod_coord_aver(rho2) == 0

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 1, 0, 0] = 4 - 2j
    rho3[2, 2, 1, 3] = 5 + 1j
    rho3[2, 2, 3, 1] = 3
    rho3[1, 1, 2, 2] = 8 - 1j
    assert prod_coord_aver(rho3) == 0.25 * (4 - 2j + (5 + 1j) * sqrt(6) + 3 * sqrt(6) + (8 - 1j) * sqrt(4))


def test_prod_impulse_aver():
    size = 4

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert prod_impulse_aver(rho1) == 0

    t = 0.761
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[1, 0, 1, 0] = 2 * t**2
    rho2[1, 0, 0, 1] = -2j * t * r
    rho2[0, 1, 1, 0] = 2j * t * r
    rho2[0, 1, 0, 1] = 2 * r ** 2
    rho2[0, 0, 0, 0] = 1
    rho2 = rho2 / 3
    assert prod_impulse_aver(rho2) == 0

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 1, 0, 0] = 4 - 2j
    rho3[2, 2, 1, 3] = 5 + 1j
    rho3[2, 2, 3, 1] = 3
    rho3[1, 1, 2, 2] = 8 - 1j
    assert prod_impulse_aver(rho3) == - 0.25 * (4 - 2j - (5 + 1j) * sqrt(6) - 3 * sqrt(6) + (8 - 1j) * sqrt(4))


def test_coord_square_aver():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert coord_square_aver(rho1, channel=1) == 0
    assert coord_square_aver(rho1, channel=2) == 0

    t = 0.379
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2
    assert cmath.isclose(coord_square_aver(rho2, channel=1), 0.75, rel_tol=1e-9)
    assert cmath.isclose(coord_square_aver(rho2, channel=2), 0.75, rel_tol=1e-9)

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3
    assert cmath.isclose(coord_square_aver(rho3, channel=1), (t**2) / 3 + 0.25, rel_tol=1e-9)
    assert cmath.isclose(coord_square_aver(rho3, channel=2), (r**2) / 3 + 0.25, rel_tol=1e-9)


def test_impulse_square_aver():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert impulse_square_aver(rho1, channel=1) == 0
    assert impulse_square_aver(rho1, channel=2) == 0

    t = 0.379
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2
    assert cmath.isclose(impulse_square_aver(rho2, channel=1), 0.75, rel_tol=1e-9)
    assert cmath.isclose(impulse_square_aver(rho2, channel=2), 0.75, rel_tol=1e-9)

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3
    assert cmath.isclose(impulse_square_aver(rho3, channel=1), (t**2) / 3 + 0.25, rel_tol=1e-9)
    assert cmath.isclose(impulse_square_aver(rho3, channel=2), (r**2) / 3 + 0.25, rel_tol=1e-9)


def test_squeezing_quadratures():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert squeezing_quadratures(rho1, channel=1) == (0, 0)
    assert squeezing_quadratures(rho1, channel=2) == (0, 0)

    t = 0.379
    r = sqrt(1 - t**2)
    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2
    sq2_1 = squeezing_quadratures(rho2, channel=1)
    sq2_2 = squeezing_quadratures(rho2, channel=2)
    assert cmath.isclose(sq2_1[0], sqrt(0.75), rel_tol=1e-9)
    assert cmath.isclose(sq2_1[1], sqrt(0.75), rel_tol=1e-9)
    assert cmath.isclose(sq2_2[0], sqrt(0.75), rel_tol=1e-9)
    assert cmath.isclose(sq2_2[1], sqrt(0.75), rel_tol=1e-9)

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3
    sq3_1 = squeezing_quadratures(rho3, channel=1)
    sq3_2 = squeezing_quadratures(rho3, channel=2)
    assert cmath.isclose(sq3_1[0], sqrt((t**2)/3 + 0.25), rel_tol=1e-9)
    assert cmath.isclose(sq3_1[1], sqrt((t**2)/3 + 0.25), rel_tol=1e-9)
    assert cmath.isclose(sq3_2[0], sqrt((r**2)/3 + 0.25), rel_tol=1e-9)
    assert cmath.isclose(sq3_2[1], sqrt((r**2)/3 + 0.25), rel_tol=1e-9)


def test_erp_squeezing_correlations():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert erp_squeezing_correlations(rho1) == (0, 0)

    t = 0.379
    r = sqrt(1 - t**2)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[2, 0, 2, 0] = 2 * (t**2 - r**2) ** 2
    rho2[2, 0, 1, 1] = - 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[2, 0, 0, 2] = 2 * (t**2 - r**2) ** 2
    rho2[1, 1, 2, 0] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[1, 1, 1, 1] = 16 * t**2 * r**2
    rho2[1, 1, 0, 2] = 4j * sqrt(2) * t * r * (t**2 - r**2)
    rho2[0, 2, 2, 0] = 2 * (t ** 2 - r ** 2) ** 2
    rho2[0, 2, 1, 1] = - 4j * sqrt(2) * t * r * (t ** 2 - r ** 2)
    rho2[0, 2, 0, 2] = 2 * (t ** 2 - r ** 2) ** 2
    rho2 = 0.25 * rho2
    epr2 = erp_squeezing_correlations(rho2)
    assert cmath.isclose(epr2[0], 3/2, rel_tol=1e-9)
    assert cmath.isclose(epr2[1], 3/2, rel_tol=1e-9)

    rho3 = np.zeros((size,) * 4, dtype=complex)
    rho3[1, 0, 1, 0] = 2 * t**2
    rho3[1, 0, 0, 1] = -2j * t * r
    rho3[0, 1, 1, 0] = 2j * t * r
    rho3[0, 1, 0, 1] = 2 * r ** 2
    rho3[0, 0, 0, 0] = 1
    rho3 = rho3 / 3
    epr3 = erp_squeezing_correlations(rho3)
    assert cmath.isclose(epr3[0], 5/6, rel_tol=1e-9)
    assert cmath.isclose(epr3[1], 5/6, rel_tol=1e-9)
