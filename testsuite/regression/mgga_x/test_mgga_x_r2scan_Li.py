
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.973315625948457e+00, -1.309794209194168e+00, -2.342832281820834e-01, -1.808641809759491e-01, -5.244988779359668e-02, -4.816589062986531e-03, -1.021140385176983e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r2scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.757525744228636e+00, -2.760062739973662e+00, -1.957398049816152e+00, -1.958904715081391e+00, -3.254056935755857e-01, -3.259033195562960e-01, -2.474647323528177e-01, 3.100805914243296e-01, -7.716512692898699e-02, 3.397468226982040e-03, 1.841838856835625e-01, 3.236239318653542e-01, 1.015185605953249e-04, -2.692620176446978e-22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.514710930372317e-04, 0.000000000000000e+00, -3.501166986341365e-04, -1.792506652403148e-03, 0.000000000000000e+00, -1.785942048356601e-03, -1.898862455824676e-02, 0.000000000000000e+00, -1.954278811610520e-02, -4.627727299367507e+00, 0.000000000000000e+00, -8.201847943569375e+03, -2.925041707262682e+01, 0.000000000000000e+00, -7.087794145405238e+06, -8.601668563391070e+01, 0.000000000000000e+00, -7.332775467524260e+03, -4.803770781743114e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.819538380577167e-02, 1.817435574935419e-02, 3.098683975490197e-02, 3.095264261266209e-02, 2.378177799997483e-03, 2.508153711235639e-03, 1.773829230416201e-01, 1.051097882968143e-01, 6.055273548559918e-02, 2.898665734182657e-03, 1.281999155765807e-03, 1.069075487569362e-01, 5.856115733943896e-10, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
