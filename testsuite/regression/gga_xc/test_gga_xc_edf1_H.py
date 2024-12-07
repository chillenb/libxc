
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_edf1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.409795695200937e-01, -5.841560608704104e-01, -3.577982025491032e-01, -1.684745785638612e-01, -1.112431431709727e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_edf1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.543219919848390e-01, -2.879660222165259e-01, -7.480087923476270e-01, -3.076716543167796e-01, -4.096206082035869e-01, -2.425341121005488e-01, -6.303279549514511e-02, -3.891206180034414e-02, -2.254743294644921e-02, -2.732296755342620e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_edf1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.869024651831574e-03, 3.928816822613601e-02, 2.945484063626384e-02, -1.433692524215532e-02, 6.244940501956699e-02, 4.677626752980434e-02, -1.472924780792945e-01, 5.375747808163538e-01, 4.031691739116190e-01, -1.818968577238907e+01, 2.353238426004574e+01, 1.764925833392812e+01, -1.005054702714365e+05, 2.017173808339469e-17, 1.512878126599881e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
