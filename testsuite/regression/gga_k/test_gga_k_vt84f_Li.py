
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vt84f_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.883462890446192e+01, 1.139396285229785e+01, 3.315073939333650e+00, 1.412178356296787e-01, 8.022109413418667e-02, 3.084472641321620e+00, 1.356897369842380e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vt84f_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.963203097730461e+01, 1.968223266373893e+01, 6.623077688321426e+00, 6.637923573152230e+00, -2.796980262097787e+00, -2.808175543103471e+00, 1.893171071882844e-01, -3.057064710239609e+00, -3.848640526762417e-02, -1.210227402334704e+00, -3.032147213773425e+00, -3.137953856729505e+00, -1.418447087985953e+00, -1.185586587718466e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vt84f_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.011322603604896e-02, 0.000000000000000e+00, 2.004738281148183e-02, 6.509127206294622e-02, 0.000000000000000e+00, 6.493518132359759e-02, 4.033053719605006e+00, 0.000000000000000e+00, 4.038985570666435e+00, 2.161838338957527e+01, 0.000000000000000e+00, 7.829811030545531e+04, 3.767872318039012e+02, 0.000000000000000e+00, 2.454376727175391e+09, 6.733474176950160e+04, 0.000000000000000e+00, 6.882556827828620e+04, 8.236143114997229e+09, 0.000000000000000e+00, 2.292358375079829e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
