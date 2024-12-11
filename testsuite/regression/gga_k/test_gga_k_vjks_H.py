
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vjks_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035499060616954e+00, 1.694650300696287e+00, 6.051827393010197e-01, -5.365227098216445e-02, -4.096648395466440e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vjks_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388527960944710e+00, -2.494963215928859e-16, 2.689169923716043e+00, 6.986436489598167e-16, 9.481421833696919e-01, 1.625353337406078e-16, 3.799406993240702e-01, -9.957462782162382e-18, 4.132134886611581e-01, 1.091728416754251e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vjks_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.090188950270006e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.282098181971038e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.321186246138025e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.283126931787698e+01, 0.000000000000000e+00, 0.000000000000000e+00, -8.757803918737625e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
