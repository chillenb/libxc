
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_1d_ehwlrg_3_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.758451972269528e-01, -2.436354479839668e-01, -1.081059643432142e-01, -1.007497950999380e-02, -3.761964751608275e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_1d_ehwlrg_3_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.723252789977033e-01, -3.723252789977033e-01, -3.431492040341359e-01, -3.431492040341359e-01, -1.693273097320709e-01, -1.693273097320709e-01, -1.621225211203831e-02, -1.621225211203831e-02, -6.056762919557382e-05, -6.056762919557382e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
