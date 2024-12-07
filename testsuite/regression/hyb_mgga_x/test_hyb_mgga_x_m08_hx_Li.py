
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m08_hx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.490724814737324e-01, -4.917208381822123e-01, -2.092524818229630e-01, -6.789801005218887e-02, -4.651659001164393e-02, 8.782188204610037e-03, 1.665204245147546e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m08_hx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.728326722266164e-01, -9.766545018674019e-01, -4.577509704099885e-01, -4.587416103036082e-01, -3.193399460350922e-01, -3.215530986531058e-01, -9.032384812655848e-02, 1.068243821847402e-02, -4.509594065453242e-02, 3.598538948686681e-04, 1.179115461182117e-02, 1.110918360245439e-02, 2.403956422240150e-04, 1.708950230787451e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.024109064707136e-04, 0.000000000000000e+00, -5.070415100755787e-04, 4.606959168753496e-04, 0.000000000000000e+00, 4.625422939773089e-04, -1.527078565115186e-02, 0.000000000000000e+00, -1.551846158732637e-02, -4.089247871031964e+01, 0.000000000000000e+00, 6.894857210472789e-01, -5.038879405261690e+01, 0.000000000000000e+00, 4.495561744179734e+00, 7.139192995573090e-01, 0.000000000000000e+00, 6.535181283415901e-01, 3.272761616089157e+00, 0.000000000000000e+00, 4.684569350195006e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.705236898375363e-02, 5.794382885845700e-02, -4.170877523162939e-02, -4.169372080612707e-02, 1.317127154294882e-02, 1.377009994945475e-02, 2.384051477334711e+00, 6.065748064944157e-05, 5.108325759998482e-02, 1.247137862615996e-08, 3.018345351189254e-08, 6.545354254684051e-05, 1.694520734402256e-19, 1.391390194448033e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
