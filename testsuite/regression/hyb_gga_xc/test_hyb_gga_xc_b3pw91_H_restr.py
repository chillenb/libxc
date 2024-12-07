
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3pw91_H_restr_1_zk():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.562027043279653e-01, -4.180093476895692e-01, -2.629158147018427e-01, -1.046716235736756e-01, -4.175649763065609e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3pw91_H_restr_1_vrho():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.954790515515259e-01, -5.281514448578918e-01, -3.132724570696022e-01, -7.694291559171369e-02, -1.074256394329419e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3pw91_H_restr_1_vsigma():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.097511970604793e-03, -8.625991828481245e-03, -6.776667837708504e-02, -6.965442617417636e+00, -3.589011759889112e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
