
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.736144422033056e+00, -1.205362289397480e+00, -2.893818409937454e-01, -1.569011844309854e-01, -6.221861459014284e-02, -1.139516090161374e-02, -2.127820881496785e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.313815416125029e+00, -2.315900226195033e+00, -1.606465181142880e+00, -1.607832512302877e+00, -3.859244636828948e-01, -3.857603409163446e-01, -2.092663575965003e-01, -1.450062406241764e-02, -8.295816684207279e-02, -4.598929182286523e-04, -1.524839211816757e-02, -1.513748933946413e-02, -3.071820507310971e-04, -2.183783727950880e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
