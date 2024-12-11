
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_b93_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_b93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.148139357430844e-01, -6.445385560087519e-01, -1.694899513600119e-01, -8.749368213870812e-02, -3.658886409720423e-02, -9.086906247965083e-03, -1.904779527432290e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_b93_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_b93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.208229093954731e+00, -1.209190551430221e+00, -8.495052874635736e-01, -8.501195991680571e-01, -2.212849198451112e-01, -2.212246238215412e-01, -1.151412985922792e-01, -6.940596035394166e-02, -4.803145324013113e-02, -3.599365699680278e-02, -1.188504730230230e-02, -1.187740169982847e-02, -2.525106199923743e-04, -2.543398328032719e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
