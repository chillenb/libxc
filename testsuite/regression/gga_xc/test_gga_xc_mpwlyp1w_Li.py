
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mpwlyp1w_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.863184697605063e+00, -1.345194848706173e+00, -4.205029921648875e-01, -1.627683355197419e-01, -8.234091954631199e-02, -4.356283453937463e-03, -4.604989430104149e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mpwlyp1w_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.310698991977647e+00, -2.312650737992905e+00, -1.593251591718343e+00, -1.594445431478922e+00, -4.577122748586604e-01, -4.580752219489920e-01, -2.075703622507017e-01, -9.341826019016999e-02, -7.618840374541445e-02, -3.559060212555062e-02, -9.849104927293129e-03, -9.624863077011799e-03, -4.154964804015498e-05, -1.164948838440626e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mpwlyp1w_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.702542712201502e-04, 4.596077571229506e-06, -2.694529486697554e-04, -9.976796876460924e-04, 3.209308774538757e-05, -9.950308954440865e-04, -7.148777105342249e-02, 4.200911319955845e-02, -7.126015263176920e-02, -4.455688884150489e+00, 4.044598597118675e+00, 3.724572895117580e+01, -7.312391824192480e+01, 2.074106966210102e+01, 4.765462317423402e+02, 3.433779449043450e+01, 6.983765643164339e-02, 3.226331381540408e+01, 3.828578645229424e+02, 0.000000000000000e+00, 5.872565727560732e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
