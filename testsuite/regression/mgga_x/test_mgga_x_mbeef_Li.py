
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbeef_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.931506215788901e+00, -1.376562251774970e+00, -3.663147655650537e-01, -1.727982080891604e-01, -8.230249851699377e-02, -1.268150442236761e-02, -2.442549378831916e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbeef_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.445620291041557e+00, -2.447949703611110e+00, -1.641981110817315e+00, -1.643511058209943e+00, -4.582164344613565e-01, -6.105718883847971e-01, -2.242602748782419e-01, -1.564257543847742e-02, -1.052959840864513e-01, -3.340273498065783e+00, -1.760739966418811e-02, -1.633328016457445e-02, -3.518763460842179e-04, -7.643679550789063e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.203374228326005e-04, 0.000000000000000e+00, -2.195939546350707e-04, -1.015375529439936e-03, 0.000000000000000e+00, -1.011707672737137e-03, -1.255429415442685e-01, 0.000000000000000e+00, 5.858753117882672e-02, -2.933385171683864e+00, 0.000000000000000e+00, 3.769355421923635e-01, -9.717907279431403e+00, 0.000000000000000e+00, 6.773170649425478e+09, 7.773473715862972e-01, 0.000000000000000e+00, 3.569003824799056e-01, 3.629055217752414e+00, 0.000000000000000e+00, 1.477874347374637e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.294951823360666e-06, -5.142946673285509e-12, 3.249775691233085e-06, 6.304065850831652e-19, 4.174260960712027e-02, 1.434470141766410e-10, -5.127857974785544e-03, -4.654108142793974e-13, 2.385752388810014e-06, -2.759629590056252e+00, -1.087388572587513e-17, -1.468502539876030e-13, 1.123251023705570e-34, -6.446960315807253e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
