
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.536437279946464e-01, -5.890822986334964e-01, -3.561770571935118e-01, -1.641097341530571e-01, -1.061596832842090e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.715187903060448e-01, -6.377190540235885e-17, -7.704081038976178e-01, -1.900468803121392e-16, -4.152073076379392e-01, 2.643657612760625e-17, -1.166465737363811e-01, -8.615035217485017e-17, -1.413477236099095e-02, -1.237631300268609e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.479857075073641e-05, 0.000000000000000e+00, 0.000000000000000e+00, -6.983587832205511e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.303708658557176e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.149984580922999e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.586349455464235e+01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
