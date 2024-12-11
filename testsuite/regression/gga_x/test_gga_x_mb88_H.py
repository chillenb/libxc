
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mb88_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.217576530816931e-01, -5.640067348569056e-01, -3.374883827494320e-01, -1.182058555394539e-01, -5.789119671059927e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mb88_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.286509506594879e-01, -1.163380578157053e-17, -7.348427814037952e-01, -1.893043523970610e-16, -4.211756259203663e-01, 3.466032800375998e-17, -9.288441544594075e-02, -5.689989402730223e-17, -1.253559974322253e-02, -2.694561352169658e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mb88_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.510479880702261e-03, 0.000000000000000e+00, 0.000000000000000e+00, -7.973526152246302e-03, 0.000000000000000e+00, 0.000000000000000e+00, -6.291669014752628e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.285261985891944e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.166253587946589e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
