
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_1d_exponential_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_exponential", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.236226525178951e-01, -3.976195826066508e-01, -8.698984266574154e-02, -2.261042597199816e-02, -2.238661983591013e-03, -2.310532556190571e-05, -3.076159389760341e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_1d_exponential_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_exponential", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.392089359909446e-01, -4.392194670451376e-01, -4.314653976851676e-01, -4.314949847800821e-01, -1.446721978142864e-01, -1.445612058953862e-01, -4.044300064259448e-02, -3.902148649420621e-05, -4.178392551819407e-03, -2.299360603521262e-09, -4.481485701093967e-05, -4.392367062281342e-05, -7.219498123720373e-10, -2.705506503987437e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
