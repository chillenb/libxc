
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_teter93_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.831087319408935e+00, -1.289838253737021e+00, -3.386583242572970e-01, -1.579233390159157e-01, -6.888712309496493e-02, -1.817678876654449e-02, -3.673115823622858e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_teter93_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.419403962313524e+00, -1.701071603131439e+00, -4.420322017589515e-01, -2.051697176887948e-01, -8.952408872040879e-02, -2.375803094839731e-02, -4.892522283378266e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
