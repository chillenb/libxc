
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bayesian_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.224540464728719e-01, -5.748821735766904e-01, -3.571774806904905e-01, -1.442005778703304e-01, -1.220913275610966e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bayesian_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.290725715557814e-01, -8.696678290827422e-17, -7.196153427550303e-01, -1.702004665549349e-16, -3.982984557144080e-01, -9.227821033400244e-18, -1.178355849542291e-01, -9.180219805995410e-17, -1.568317835282261e-02, -1.003639260811607e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bayesian_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.328572110082745e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.178189350876092e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.702118200454363e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.378045821829577e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.759833431939701e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
