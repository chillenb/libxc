
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035441295179471e+00, 1.698003380543957e+00, 6.201244442147773e-01, 6.233563777074969e-02, 1.611887382583046e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388579043981290e+00, 4.560935815377587e-16, 2.670580793931621e+00, 2.012504579959150e-16, 8.930950954793493e-01, 1.676501670676255e-16, 8.504752942213178e-02, 3.431969855631443e-17, 2.685012923089512e-04, -2.858505061251143e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.864159801331974e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.405123158282043e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.067239603922496e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.121214857383630e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.171512014607462e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
