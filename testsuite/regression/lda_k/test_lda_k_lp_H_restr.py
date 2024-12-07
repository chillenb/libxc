
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_lp_H_restr_1_zk():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.324803692481525e+00, 1.065108343100062e+00, 3.643771332423766e-01, 2.598725040171233e-02, 5.767611994114802e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_lp_H_restr_1_vrho():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.208006154135874e+00, 1.775180571833436e+00, 6.072952220706278e-01, 4.331208400285389e-02, 9.612686656858003e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
