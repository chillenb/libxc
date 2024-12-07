
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_rpa_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.152512401108542e-01, -1.048586074252658e-01, -6.730951127383017e-02, -3.469060638333209e-02, -2.425245616406194e-02, -1.440994854205462e-02, -1.017935336578130e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_rpa_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.249299327742583e-01, -1.247705075875385e-01, -1.143079629428460e-01, -1.141721495925486e-01, -7.530557550886290e-02, -7.534647152473846e-02, -3.872940131649248e-02, -1.327478948740958e-01, -2.765666507928350e-02, -7.945832668557300e-02, -1.716718766531270e-02, -1.725364472795377e-02, -1.232589393304446e-03, -1.319698160899579e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
