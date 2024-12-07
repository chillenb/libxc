
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pmgb06_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.343166352528429e-03, -4.425384667603202e-03, -1.171215722918517e-02, -1.706799624178875e-02, -1.706864039608201e-02, -6.746735053667639e-03, -1.789528921482774e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pmgb06_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.474186945293064e-03, -3.308332190032119e-03, -9.572713293182981e-03, -1.517867789872916e-02, -1.909202271878484e-02, -8.504879892813370e-03, -2.365938515785235e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
