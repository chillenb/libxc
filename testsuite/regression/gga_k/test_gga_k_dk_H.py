
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_dk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035148122468553e+00, 1.692941560273740e+00, 6.216137829616845e-01, 1.119710468856254e-01, 6.800877245328337e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_dk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388861620419375e+00, -3.836284885754024e-16, 2.657857327581334e+00, 8.419954480870812e-16, 8.925028333736217e-01, 2.553261400806052e-16, -3.094160219577081e-01, 2.886993338033054e-22, -6.887166681586293e-01, -9.033324124691092e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_dk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.681243119146274e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.604254596980901e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.134384496707806e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.583361275231591e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.456077427594465e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
