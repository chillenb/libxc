
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_acgga_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.206817313159336e-02, -1.873151326069743e-02, -8.715842745126247e-03, -2.045328241275998e-04, -4.143335607335180e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_acgga_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.696699333450063e-02, 8.365304346476996e+00, -4.080662940701670e-02, 3.375966063381846e+02, -2.811241382864949e-02, 1.983959906733108e+02, -1.165011243132362e-03, 2.109612759319283e+00, -2.626091486629401e-09, 1.227406054603915e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_acgga_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.617350487958958e-02, 3.234700975917917e-02, 1.617350487958958e-02, 9.720432027288154e-03, 1.944086405457631e-02, 9.720432027288154e-03, 4.155009112869910e-02, 8.310018225739819e-02, 4.155009112869910e-02, 1.072966063588585e-01, 2.145932127177169e-01, 1.072966063588585e-01, 1.702075919883049e-03, 3.404151839766098e-03, 1.702075919883049e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
