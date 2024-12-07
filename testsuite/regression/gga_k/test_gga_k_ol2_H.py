
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ol2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.044200664261064e+00, 1.698973586774937e+00, 6.189904499888259e-01, 9.669711286205965e-02, 7.620018025399170e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ol2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.398849907207840e+00, 1.751770616762197e-16, 2.691127694710291e+00, 2.995713997494799e-16, 8.834769793116717e-01, 3.349889435915157e-17, 1.050405672683942e-02, 5.170499823707928e-17, -7.596222121313126e-02, -1.817939362256802e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ol2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.250287870284823e-01, 0.000000000000000e+00, 0.000000000000000e+00, 6.525848827713167e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.236016738386260e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.695803473925528e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.621829337742967e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
