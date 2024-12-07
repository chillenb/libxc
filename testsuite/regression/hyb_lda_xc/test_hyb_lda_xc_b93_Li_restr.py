
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_b93_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_b93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.148136110274390e-01, -6.445383600171813e-01, -1.694899417421453e-01, -7.894591359777978e-02, -3.443572995656714e-02, -9.086885320177476e-03, -1.907558647663549e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_b93_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_b93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.208710032407416e+00, -8.498125686491349e-01, -2.212547775345164e-01, -1.026015172223344e-01, -4.473768746942190e-02, -1.188123299878948e-02, -2.533362107223259e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
