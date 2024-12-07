
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_lda0_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_lda0", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.372220416541158e+00, -9.668075400257718e-01, -2.542349126132180e-01, -1.184188703966697e-01, -5.165359493485071e-02, -1.363032798026621e-02, -2.861337971495322e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_lda0_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_lda0", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.813065048611125e+00, -1.274718852973702e+00, -3.318821663017745e-01, -1.539022758335016e-01, -6.710653120413285e-02, -1.782184949818422e-02, -3.800043160834888e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
