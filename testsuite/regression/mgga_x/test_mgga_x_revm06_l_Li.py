
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revm06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.010007663856622e+00, -1.467708675268881e+00, -2.467515779048023e-01, -1.618442233650095e-01, -6.913697757750084e-02, -9.130918937583471e-03, -1.708346434794514e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revm06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.964592439186940e+00, -1.963105357543862e+00, -1.858216564730317e+00, -1.859120136883773e+00, -4.783117808560656e-01, -4.854997630987214e-01, -1.477356809178223e-01, -1.155353009732829e-02, -9.682876806298012e-02, -3.692236649071312e-04, -1.223090889406511e-02, -1.205530061474284e-02, -2.466245939858694e-04, -1.753270514387001e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revm06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.629889409840462e-04, 0.000000000000000e+00, -4.613140914245582e-04, -1.824891861051433e-03, 0.000000000000000e+00, -1.819413904015205e-03, -6.387126176214114e-02, 0.000000000000000e+00, -6.519007383506091e-02, -6.790201913591207e+00, 0.000000000000000e+00, 7.447355217127062e-02, -8.801936970692380e+01, 0.000000000000000e+00, 4.838172701340661e-01, -1.098001461164201e-01, 0.000000000000000e+00, 7.060474567509273e-02, -5.755296878547592e-01, 0.000000000000000e+00, 5.041552863447440e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revm06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.321488722246206e-02, -6.368620163136433e-02, 3.634909978206793e-02, 3.622760396285703e-02, 5.156208319497554e-02, 5.370668446820206e-02, -1.541094999330335e+00, -1.089673834831122e-05, 3.756956697014068e-01, -2.238961072842022e-09, 1.872618183365738e-08, -1.175878292231580e-05, 1.623332167712285e-19, -2.497931211677958e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
