
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_llp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035650574038136e+00, 1.700807445257947e+00, 6.189032154292039e-01, 6.673262414284284e-02, 1.254884476020895e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_llp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388385660880465e+00, 1.458615405578182e-16, 2.677588621791846e+00, 2.406619895861688e-16, 9.019227838475665e-01, 1.226853129967601e-16, 7.010788613590331e-02, 3.040656928878490e-17, 7.407470898139474e-04, -2.095072023274415e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_llp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.695795124873767e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.296693250127456e-02, 0.000000000000000e+00, 0.000000000000000e+00, 2.829997377125010e-01, 0.000000000000000e+00, 0.000000000000000e+00, 4.627694713194467e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.079336589655350e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
