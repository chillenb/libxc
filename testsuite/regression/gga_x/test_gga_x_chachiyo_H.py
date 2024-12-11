
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_chachiyo_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.221603087702338e-01, -5.797307337689652e-01, -3.608720717339752e-01, -1.548336621740331e-01, -8.273537407030712e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_chachiyo_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.281386310693509e-01, 1.946495311414801e-18, -7.197774395762869e-01, -2.315769328756017e-16, -4.035268421943612e-01, -6.052032138283361e-17, -8.870777890929003e-02, -6.841803940968695e-17, -2.054872636622185e-02, -1.511258722176309e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_chachiyo_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.160374652491721e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.470941096068070e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.695516931674820e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.325248294801357e+01, 0.000000000000000e+00, 0.000000000000000e+00, -7.172934284915928e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
