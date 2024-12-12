
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_rda_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.611150581965515e+00, 4.813913543931757e-01, 4.545026194153997e-01, 3.548123800413722e-01, 6.841342994215921e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_rda_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.812471728063989e+00, -1.694989870815785e-15, 4.117127910640492e+00, 9.189968532612381e-17, 1.375626534147655e-01, 8.094744762312823e-17, -3.744295984354014e-01, 1.711119831564690e-16, -6.864154652924562e-01, -4.112890122077982e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.613722321706782e-01, 0.000000000000000e+00, 0.000000000000000e+00, 9.835715221082778e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.279879310272941e+00, 0.000000000000000e+00, 0.000000000000000e+00, 9.041167145722245e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.459609669870486e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([5.878385448581279e-02, 0.000000000000000e+00, 1.394447694931417e-01, 0.000000000000000e+00, 1.694897266263004e-01, 0.000000000000000e+00, 3.656820290003202e-02, 0.000000000000000e+00, 2.819025683648352e-06, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
