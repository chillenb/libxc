
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt_ge_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547496940525e-01, -5.689813530268767e-01, -3.461995181540083e-01, -1.288398668656890e-01, -4.104593376403017e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt_ge_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285213103564698e-01, -6.837474302615595e-17, -7.283463139972727e-01, -1.854909171797702e-16, -4.111350170128794e-01, 3.463051285941670e-17, -1.158004720609810e-01, -5.992819397918798e-17, -5.531064750782905e-03, -2.871255033691295e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt_ge_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.484773949675210e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.407194979416160e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.102107680023455e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.301783479604086e+00, 0.000000000000000e+00, 0.000000000000000e+00, 4.656509990260279e+01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
