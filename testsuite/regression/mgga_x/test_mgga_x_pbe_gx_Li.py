
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_pbe_gx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.119164116982275e+00, -1.456779734439053e+00, -2.762254005967172e-01, -1.920612823000739e-01, -6.785693839428888e-02, -7.018959225327935e-05, -9.009913532723338e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_pbe_gx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.791395069923126e+00, -2.794008046915960e+00, -1.895318122052807e+00, -1.896991924020990e+00, -2.055497279296861e-01, -2.132830750918389e-01, -2.548252197533284e-01, 5.593281331022330e-02, -7.405974864220354e-02, 1.800559177085994e-03, -2.365568981329519e-04, 5.835094683694302e-02, -4.176439973008035e-09, 1.559889221970741e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pbe_gx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.077282082095598e-04, 0.000000000000000e+00, -3.066192922066075e-04, -1.315393622640465e-03, 0.000000000000000e+00, -1.310808682240352e-03, -3.474114053854199e-01, 0.000000000000000e+00, -3.364798223512440e-01, -4.617552222584218e+00, 0.000000000000000e+00, -1.438178141888416e+03, -1.718294939158060e+02, 0.000000000000000e+00, -3.651622923077556e+06, 1.310621762955518e+00, 0.000000000000000e+00, -1.285154557990800e+03, 6.062562472335664e+00, 0.000000000000000e+00, -3.016111242018797e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pbe_gx_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pbe_gx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.086739814880790e-02, 2.084876870966884e-02, 2.976190858745893e-02, 2.973565841958481e-02, 1.016204975571006e-01, 9.947265497243082e-02, 2.318092027832028e-01, 1.839183288016973e-02, 5.200294807098370e-01, 1.487805368542436e-03, 6.731926909396663e-13, 1.869836983336668e-02, 3.158395172370137e-27, 1.315728375387277e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
