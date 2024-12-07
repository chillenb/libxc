
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_hcth_a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.609140258547082e-02, -5.913738465214774e-02, -1.172917668722112e-02, -2.617257324312230e-03, -7.561194445493260e-03, 1.298172851149942e-02, 1.881711414417168e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_hcth_a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.778034170917862e-02, -2.765833997055726e-02, -3.050796220898382e-02, -3.038315589327009e-02, -1.229081627556749e-01, -1.234815484723071e-01, 3.488741468839783e-03, 5.804866014200637e-01, -6.816044220251499e-03, 3.559784589100392e-01, 1.624634992956355e-02, 1.681757050703352e-02, 7.355984056369148e-05, 7.494380965549031e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_hcth_a_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.784448442182579e-05, 0.000000000000000e+00, -5.759652400861849e-05, -1.835632451785063e-04, 0.000000000000000e+00, -1.828428954724667e-04, 5.310962986464728e-02, 0.000000000000000e+00, 5.330149499968992e-02, -3.272190776344972e+00, 0.000000000000000e+00, 1.058998480647012e+02, -4.851175317215666e+00, 0.000000000000000e+00, 1.252526435830120e+04, 1.240088227759239e+00, 0.000000000000000e+00, 1.315357076447953e+00, 1.955533582044776e+00, 0.000000000000000e+00, 3.054729295759348e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
