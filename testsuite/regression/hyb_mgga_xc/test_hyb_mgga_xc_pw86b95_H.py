
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pw86b95_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw86b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.414710219936537e-01, -4.090038479540188e-01, -2.600520530125660e-01, -9.741757175133972e-02, -1.417006168514767e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pw86b95_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw86b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.884630697970628e-01, -2.638962389450027e-01, -5.030309643233686e-01, -2.439192485430376e-01, -2.802297999740531e-01, -1.780651619035293e-01, -9.427793354486429e-02, -3.716537509663862e-02, -1.137358512531391e-02, -8.062502798546464e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw86b95_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw86b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.595388226821465e-03, 0.000000000000000e+00, 1.737704374687904e+20, -1.610416623421240e-02, 0.000000000000000e+00, 1.557037836188098e+20, -1.288894512814968e-01, 0.000000000000000e+00, 1.069289566183098e+20, -3.917349927452884e+00, 0.000000000000000e+00, 1.012426419284584e+19, -6.008792236991171e+03, 0.000000000000000e+00, 6.156766463398428e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw86b95_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw86b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.574338818834315e-02, 0.000000000000000e+00, -9.757973543600244e-03, 0.000000000000000e+00, -9.002796676653201e-03, 0.000000000000000e+00, -9.556339337177224e-04, 0.000000000000000e+00, -1.542838856662459e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
