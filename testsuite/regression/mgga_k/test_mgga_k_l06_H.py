
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_l06_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.910090571975003e+00, 1.781351617888169e+00, 6.199561627629673e-01, 8.494908501536760e-02, 1.988870539163835e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_l06_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.912553876456433e+00, 6.492064554731599e-16, 2.515615410798153e+00, 1.731179471407491e-16, 8.563015098416196e-01, 1.474995759881680e-16, 1.177595781332572e-01, 3.101359480369221e-17, 3.314783186967099e-04, -5.879154354760612e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.181379759145290e-01, 0.000000000000000e+00, 0.000000000000000e+00, 9.302382676424831e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.883283676131007e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.485524073478602e+00, 0.000000000000000e+00, 0.000000000000000e+00, 6.042643518281037e-05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-5.488429962756425e-03, 0.000000000000000e+00, -1.000524924021103e-02, 0.000000000000000e+00, -4.263297542831846e-03, 0.000000000000000e+00, 3.915364281157225e-04, 0.000000000000000e+00, 4.089617223098434e-12, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
