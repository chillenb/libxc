
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_lp96_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([3.357805039488487e-02, 3.177688683823361e-02, 1.473735034560064e-02, -7.924334934337789e-03, -2.485359008960916e-02, 1.026711418935192e+00, 5.148316432340722e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_lp96_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.495184125342805e-02, 3.495184125342805e-02, 3.372578462249905e-02, 3.372578462249905e-02, 2.156811944397910e-02, 2.156811944397910e-02, 2.738517136871842e-03, 2.738517136871842e-03, -3.305670715261904e-02, -3.305670715261904e-02, 1.509347206732278e-01, 1.509347206732278e-01, 1.703952524396781e+03, 1.703952524396781e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
