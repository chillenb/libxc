
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lag_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.216353242076692e-01, -5.596372082578959e-01, -3.317525693767990e-01, -1.167179757816419e-01, -4.004003023713718e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lag_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.288031619028664e-01, -7.260752064924982e-17, -7.383097940640566e-01, -1.569238160321171e-16, -4.227291971180691e-01, -1.810357323952175e-17, -9.078980852164603e-02, -6.455581500302513e-17, -1.031975642393380e-02, -1.348603141700820e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lag_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.739369632986129e-04, 0.000000000000000e+00, 0.000000000000000e+00, -3.656998956554656e-03, 0.000000000000000e+00, 0.000000000000000e+00, -4.282161323776513e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.297729942085027e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.441386313153195e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
