
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_vt84_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.139837623652237e-01, -6.417594165626792e-01, -3.739772079504660e-01, -1.260167033339737e-01, -3.680868044607663e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_vt84_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.236744862478841e-01, -6.165901489960681e-17, -7.580103444404303e-01, -3.095452438253075e-17, -4.577570269541589e-01, -1.006314717895877e-16, -1.045844676241464e-01, -2.593592174105750e-17, -5.940659847061469e-03, -8.635069143175516e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.241315556251291e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.219682544691716e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.520201458140993e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.344673425868161e+00, 0.000000000000000e+00, 0.000000000000000e+00, 8.253247031338241e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.249083765418780e+01, 0.000000000000000e+00, 2.107217246104242e-01, 0.000000000000000e+00, 8.957898381674018e-02, 0.000000000000000e+00, 2.139924243740563e-03, 0.000000000000000e+00, -9.391683472492631e-09, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
