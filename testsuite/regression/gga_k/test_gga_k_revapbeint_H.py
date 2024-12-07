
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_revapbeint_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035146516278514e+00, 1.689370447056879e+00, 6.172588008858213e-01, 6.924560679734054e-02, 1.986749209133277e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_revapbeint_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388867504507625e+00, 1.137730491967779e-16, 2.668592468173264e+00, 3.273640380758853e-16, 8.783726083474352e-01, 3.154391650060659e-17, 8.289044957070080e-02, 3.117957155645212e-17, 3.307956178276799e-04, -3.013117801324186e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_revapbeint_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.668111550414335e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.829160452691267e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.284462699567732e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.660325318758115e+00, 0.000000000000000e+00, 0.000000000000000e+00, 2.630998856424867e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
