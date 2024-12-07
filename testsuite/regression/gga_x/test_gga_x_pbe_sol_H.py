
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_sol_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547296822704e-01, -5.689258149041525e-01, -3.459154488031788e-01, -1.248640798460954e-01, -7.394595386584036e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_sol_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285213903575762e-01, -7.176302258019733e-17, -7.285613523852864e-01, -2.113465496974866e-16, -4.121634146541586e-01, 2.126393814898106e-17, -1.201191064118644e-01, -6.249797082389207e-17, -9.847156439231397e-03, -9.957587230281206e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_sol_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.483137564039013e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.393767076579966e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.071376310637516e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.218992328669615e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.832011443865875e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
