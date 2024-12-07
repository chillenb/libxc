
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_lp_a_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [4.391032653132906e-17, 3.837384909484512e-17, -1.879361231238235e-17, -3.177166308855790e-18, -1.083869002431549e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_lp_a_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.111292746415603e-16, -2.304984323824018e+00, 2.484157187464112e-16, -2.066755511425071e+00, -2.505814974984313e-17, -1.208837067088183e+00, -4.236221745141053e-18, -3.228288779943940e-01, -1.357493274356828e-18, -1.520863472487704e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
