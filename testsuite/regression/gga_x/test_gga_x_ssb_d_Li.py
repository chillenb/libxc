
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ssb_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.882744714255456e+00, -1.341446861531542e+00, -4.379162486170957e-01, -1.674803958659026e-01, -8.198138082369380e-02, -2.054643184354645e-02, -3.838586620717250e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ssb_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.436899285180764e+00, -2.439398114114784e+00, -1.531438274499287e+00, -1.533135487244578e+00, -4.130526804388020e-01, -4.132441388382551e-01, -2.276011930624341e-01, -2.611913575899812e-02, -7.732467968109337e-02, -8.296434795043973e-04, -2.746122741616271e-02, -2.726371534534610e-02, -5.541555211040217e-04, -3.939541765597661e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ssb_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.246640699750898e-04, 0.000000000000000e+00, -1.238099977112558e-04, -1.363859810434372e-03, 0.000000000000000e+00, -1.357944307357459e-03, -9.184690678448387e-02, 0.000000000000000e+00, -9.165574274719743e-02, 2.046870809733782e+00, 0.000000000000000e+00, -2.774606284782473e-01, -6.999886344952694e+01, 0.000000000000000e+00, -1.658543484838068e+00, -2.816115670261720e-01, 0.000000000000000e+00, -2.643996407308501e-01, -1.213215320808353e+00, 0.000000000000000e+00, -1.727304975190035e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
