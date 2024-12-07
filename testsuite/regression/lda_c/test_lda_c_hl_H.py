
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_hl_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_hl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.410118897517374e-02, -6.180485364433048e-02, -5.079059934996375e-02, -2.693313798464079e-02, -2.379040148130322e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_hl_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_hl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.114155862653902e-02, -7.114155862653899e-02, -6.879650754307588e-02, -6.879650754307588e-02, -5.746753794323608e-02, -5.746753794323609e-02, -3.211337611480863e-02, -3.211337611480863e-02, -3.129215637981557e-03, -3.129215637981557e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
