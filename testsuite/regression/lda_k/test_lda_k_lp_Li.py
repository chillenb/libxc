
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_lp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.640424095866549e+01, 7.907143709893439e+00, 4.557502580712749e-01, 1.340176305980224e-01, 2.106808682098142e-02, 7.066934417818417e-04, 2.511023042810305e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_lp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.731574502254819e+01, 2.736499164421084e+01, 1.316734648058948e+01, 1.318977060205576e+01, 7.599066554070941e-01, 7.592604591960590e-01, 2.234366750488754e-01, 1.072825323470195e-03, 3.511348401706296e-02, 1.079118339790408e-06, 1.186325136435063e-03, 1.169131414594245e-03, 4.814456868931332e-07, 2.433183601043695e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
