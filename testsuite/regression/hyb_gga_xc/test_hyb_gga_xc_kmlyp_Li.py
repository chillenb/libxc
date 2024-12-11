
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_kmlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.576961102837785e-01, -6.142278867112484e-01, -1.593748843834340e-01, -8.873324929106743e-02, -4.102601688006021e-02, -1.370457472670080e-02, -4.211190066496969e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_kmlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.122724824200272e+00, -1.123460183826715e+00, -8.038717957426765e-01, -8.043070892595156e-01, -2.596176685565496e-01, -2.597621147015016e-01, -1.141567300360214e-01, -1.124782046960892e-01, -5.210117230255913e-02, -5.691814053139209e-02, -1.731084190178005e-02, -1.734463390467033e-02, -5.692522741761974e-04, -5.044724819634328e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_kmlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.957435927270960e-06, 2.339821308989567e-06, 2.901851252737517e-06, 1.815892033595159e-05, 1.633829921583367e-05, 1.786580187794208e-05, 1.550562219000944e-02, 2.138645762886612e-02, 1.557658298345459e-02, -1.879228365220508e-04, 2.059068376714962e+00, 1.544949301957657e+00, 9.759345905171678e-07, 1.055909000979688e+01, 7.919318379140952e+00, 1.830946221589782e-02, 3.555371600156391e-02, 1.840351479782647e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
