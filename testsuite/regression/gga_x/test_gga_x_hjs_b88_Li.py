
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.717954109618783e+00, -1.200796887441963e+00, -3.340577015956990e-01, -1.052253100260016e-01, -2.668766095947228e-02, -1.183177689407661e-01, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.202603328149042e+00, -2.204716194162764e+00, -1.486339664860590e+00, -1.487705510027878e+00, -2.783960075942549e-01, -2.782358356665569e-01, -1.513440895956331e-01, 2.878270543565051e-01, -2.988470436184119e-02, 2.736247962713068e-18, 1.764467512837097e-01, 2.257269509173557e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.841317134702886e-04, 0.000000000000000e+00, -1.835124216293197e-04, -7.085454537475069e-04, 0.000000000000000e+00, -7.062983049082914e-04, -9.073699699435836e-02, 0.000000000000000e+00, -9.072558411681471e-02, -1.992837452918304e+00, 0.000000000000000e+00, -4.521667352607999e+03, -3.480816929728348e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.904496060746623e+03, 0.000000000000000e+00, -3.337044993092453e+03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
