
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
    ref_tgt = [-7.139837625623933e-01, -5.618443601648737e-01, -3.260643818321831e-01, -8.985097572086821e-02, -4.245996226544091e-03]
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
    ref_tgt = [-8.236744862413153e-01, -1.287436871980336e-16, -7.546844216553229e-01, -2.878585878317090e-17, -4.363673592062575e-01, -1.123497940765715e-16, -1.192641350762087e-01, -1.247151109363502e-17, -5.661298918055194e-03, -1.496107038196708e-18]
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
    ref_tgt = [-5.241315559236527e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.024776457232669e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.353831310417716e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.533226263148133e+00, 0.000000000000000e+00, 0.000000000000000e+00, 6.480088910157694e+04, 0.000000000000000e+00, 0.000000000000000e+00]
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
    ref_tgt = [1.249083768058259e+01, 0.000000000000000e+00, 7.102500920221372e-03, 0.000000000000000e+00, 1.941228147722336e-03, 0.000000000000000e+00, -6.338859030830383e-04, 0.000000000000000e+00, -2.631692126745438e-08, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
