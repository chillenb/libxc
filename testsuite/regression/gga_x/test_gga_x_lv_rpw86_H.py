
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lv_rpw86_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218001155456225e-01, -5.664288214758859e-01, -3.424059861944526e-01, -1.381627413277570e-01, -1.969758250338571e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lv_rpw86_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285939224027405e-01, -9.567176417111014e-17, -7.311288355591808e-01, -2.800216303510097e-16, -4.129986553220240e-01, 2.330895616006498e-17, -1.196238516967727e-01, -5.064639894749758e-17, -1.583649350337168e-02, -1.368466877415612e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lv_rpw86_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.253644212104000e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.119866299841726e-02, 0.000000000000000e+00, 0.000000000000000e+00, -9.509427658841223e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.270600466669885e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.331948865338716e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
