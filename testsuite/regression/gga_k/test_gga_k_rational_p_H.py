
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_rational_p_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.029287667450277e+00, 1.440374980720316e+00, 3.888924434448026e-01, 3.657889985099205e-03, 7.628839309315753e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_rational_p_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.394705050354925e+00, 7.609864417970051e-16, 2.868056352706661e+00, 1.708658727438832e-16, 9.830267830698375e-01, 7.435398124295255e-17, 1.775315734563766e-02, 1.781062831198412e-18, 4.321726584387665e-09, -1.215634687371131e-25])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_rational_p_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.926377462407248e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.171171673177364e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.313397587389660e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.312074687357592e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.437391049681562e-03, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
