
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_b97x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.701269878053984e+00, -1.181603596169000e+00, -3.437571773196024e-01, -1.045518581483958e-01, -2.626833083101795e-02, -1.030757926929543e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_b97x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.216032011859357e+00, -2.218151726499461e+00, -1.493215225756754e+00, -1.494600460341909e+00, -2.276465861944928e-01, -2.275236918614496e-01, -1.519298098246764e-01, -2.050986802988531e-03, -2.658442081466818e-02, -1.567825039724153e-17, -2.519422737512666e-03, -2.447649915887376e-03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_b97x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.228404023893072e-04, 0.000000000000000e+00, -1.223944863770027e-04, -5.369554796436635e-04, 0.000000000000000e+00, -5.351187394021523e-04, -1.216147732180289e-01, 0.000000000000000e+00, -1.215892141100897e-01, -1.241329176696889e+00, 0.000000000000000e+00, -1.819332755576935e-01, -4.041855319365601e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.171827951984658e-01, 0.000000000000000e+00, -1.983073804439795e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
