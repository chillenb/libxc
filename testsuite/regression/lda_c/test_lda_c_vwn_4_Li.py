
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.400975056368682e-02, -8.420806949892468e-02, -4.968925359667797e-02, -1.806122719511515e-02, -1.097300190848972e-02, -6.795037222377569e-03, -1.639319040398701e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.031820842229149e-01, -1.030224490508076e-01, -9.308400104767572e-02, -9.294798016063320e-02, -5.684255166381888e-02, -5.688349701702567e-02, -2.097206776083119e-02, -1.337441439071608e-01, -1.310880063386405e-02, -7.814377832487043e-02, -8.550029715130457e-03, -8.636744538116452e-03, -1.948323895342408e-04, -2.788217429793010e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
