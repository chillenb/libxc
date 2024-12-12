
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbrxh_bg_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.770729555860147e-01, -5.571341645598572e-01, -4.279191877427253e-01, -4.247967326983121e-01, -3.911455574365987e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbrxh_bg_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.684804192967851e-01, -9.810306805793008e-17, -5.794157452779709e-01, 1.189972256316702e-16, -2.364696344892150e-01, -1.150107185504420e-16, 1.919302217050916e-01, -1.414558857525610e-16, 4.114118025717267e+00, -1.127899425707118e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.039141820521414e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.416448094686791e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.728450578501641e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.473840503038942e+01, 0.000000000000000e+00, 0.000000000000000e+00, -7.454871523916311e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.596310327173696e-02, 0.000000000000000e+00, -3.231443476646551e-02, 0.000000000000000e+00, -3.125568186273035e-02, 0.000000000000000e+00, -6.486528691067302e-03, 0.000000000000000e+00, -1.000656519495816e-04, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
