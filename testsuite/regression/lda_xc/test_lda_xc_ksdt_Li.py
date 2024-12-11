
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_ksdt_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.827033387946117e+00, -1.286785602722369e+00, -3.387307479952412e-01, -1.749552286887414e-01, -7.322549596417181e-02, -1.822973588705205e-02, -3.731908926994195e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_ksdt_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.413773593518238e+00, -2.415681368877868e+00, -1.696305569264714e+00, -1.697523741935915e+00, -4.420068044286014e-01, -4.418871893175662e-01, -2.301483484272209e-01, -1.529910683918144e-01, -9.613112976194724e-02, -7.872422319009464e-02, -2.385717321314181e-02, -2.384881664696317e-02, -4.922362494140681e-04, -5.068563141492211e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
