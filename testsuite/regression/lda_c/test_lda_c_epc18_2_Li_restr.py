
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc18_2_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.321170894268232e-02, -8.142401717977499e-02, -3.975053648394695e-03, -3.084498901905819e-04, -1.916540984434302e-05, -2.354223680143820e-07, -1.322429895106192e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc18_2_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.016170013624494e-03, 5.454795767596939e-03, -8.069215996540846e-03, -6.176574037682654e-04, -3.833375734887861e-05, -4.708451794167316e-07, -2.644859790226376e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
