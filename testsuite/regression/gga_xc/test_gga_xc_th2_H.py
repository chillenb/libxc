
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.533225887886741e-01, -6.038327326144748e-01, -3.653264786855663e-01, -1.359068875817646e-01, 4.648285585505861e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.147200819156054e-01, -8.870786006498782e-02, -7.434106887825784e-01, -3.052412944167049e-02, -3.737127457968664e-01, -3.937002194988153e-02, -9.417156702478156e-02, 9.606118283435276e-02, 4.415543224783228e-03, 3.763961493329049e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.190036378659196e-02, -1.338309700283061e-02, 3.800379632142411e+17, -4.219617311456063e-02, -1.895747546521504e-02, 2.252419153172555e+17, -2.807851873950539e-01, -1.882064731254028e-01, -3.945246231748470e+17, -1.139810632443437e+01, -3.218986488460253e+01, -1.090332487699559e+18, 3.414515133160385e+04, -6.177738508832077e+05, -9.084074554850070e+17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
