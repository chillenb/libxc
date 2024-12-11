
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
    ref_tgt = [-7.015716599791937e-01, -5.618348690112902e-01, -3.260643686547126e-01, -8.982900294918848e-02, -4.244748943215425e-03]
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
    ref_tgt = [-8.297048133077563e-01, -3.235620234650351e-17, -7.546492117445536e-01, -1.595670007580052e-16, -4.363669027978848e-01, 5.195819125659339e-18, -1.192431700703007e-01, -3.917591934104399e-17, -5.659635789854343e-03, -6.844823294546197e-19]
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
    ref_tgt = [-4.322980847144798e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.057563629176050e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.354321000066063e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.426106090306997e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.842070012224491e+05, 0.000000000000000e+00, 0.000000000000000e+00]
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
    ref_tgt = [1.030813777494461e+01, 0.000000000000000e+00, 7.073681261703720e-03, 0.000000000000000e+00, 1.940700615598679e-03, 0.000000000000000e+00, -6.240536466219934e-04, 0.000000000000000e+00, -2.586912629106759e-08, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
