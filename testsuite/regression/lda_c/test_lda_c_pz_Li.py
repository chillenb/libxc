
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pz_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.319543761172203e-02, -8.322771992503267e-02, -4.988265046702828e-02, -1.819420689433543e-02, -1.097517924953430e-02, -6.749632231591079e-03, -1.672862342604614e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pz_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.025851793630207e-01, -1.023815904882874e-01, -9.221858120137043e-02, -9.204748180236270e-02, -5.679501609614435e-02, -5.684588138561322e-02, -2.117055756906108e-02, -1.135371176124165e-01, -1.314311438931329e-02, -6.788532981601249e-02, -8.488187635483025e-03, -8.585925134371251e-03, -1.994023115233301e-04, -2.814219358193011e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
