
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_yukawa_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.599089813297383e+00, -1.072458605571525e+00, -1.855466036850428e-01, -4.887463351105124e-02, -7.008086520661782e-03, -1.244637537648536e-04, -7.201126686411551e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_yukawa_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.174472666987114e+00, -1.469962250178734e+00, -2.723809686502299e-01, -7.815104536963728e-02, -1.264093227766070e-02, -2.465402732628289e-04, -1.440220729212763e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
