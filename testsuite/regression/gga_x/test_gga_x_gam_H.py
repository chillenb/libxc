
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_gam_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.931098496765404e-01, -6.277240521140338e-01, -3.757419634409431e-01, -1.448392891003955e-01, -2.060811846471990e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_gam_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.262290461643089e-01, -3.396100227448427e-17, -8.174041586629178e-01, -1.521113105710050e-16, -4.839653290612437e-01, -1.645655252949170e-16, 3.386431199884647e-02, -2.346874827744094e-17, -2.728074648312662e-02, -4.366829184133474e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_gam_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.593189874803089e-03, 0.000000000000000e+00, 0.000000000000000e+00, -8.397656185744635e-03, 0.000000000000000e+00, 0.000000000000000e+00, -3.747210480878001e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.582568846262774e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.171154359400738e+02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
