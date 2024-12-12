
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_modtpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.015716598165702e-01, -6.293270764165335e-01, -3.695723353416603e-01, -1.334274709817873e-01, -7.396949790171973e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_modtpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.297048133056346e-01, -7.864169352568259e-17, -7.720939703377654e-01, -1.988625828338113e-16, -4.581915057588912e-01, -2.132943239544723e-17, -1.291397560771698e-01, -5.359782553174056e-17, -9.856558565499838e-03, -7.154536781108018e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.322980844941712e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.291106908704150e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.536490661407977e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.518733732185287e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.827482646263598e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.030813775372191e+01, 0.000000000000000e+00, 1.424566201978404e-01, 0.000000000000000e+00, 4.301471851725745e-02, 0.000000000000000e+00, 3.135924904646140e-04, 0.000000000000000e+00, 3.879179055207894e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
