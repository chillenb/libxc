
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_lp_b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.025080255824628e+00, -1.427410335156581e+00, -3.519563486289996e-01, -2.025040678013933e-04, -4.122254572956769e-08, -1.397180983871876e-02, -1.932726656087427e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_lp_b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.672411595035355e+00, -2.661468599504322e+00, -1.890379483531848e+00, -1.883092996520426e+00, -4.678242298353369e-01, -4.687224836173843e-01, -6.731929903021060e-05, -6.087232654206538e-01, -1.372068431163764e-08, -2.419591818291562e-01, -1.832485802521465e-02, -1.893684353676155e-02, -1.338641683423495e-04, -6.023579215774634e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
