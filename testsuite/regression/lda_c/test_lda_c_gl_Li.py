
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_gl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_gl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.154324713256673e-01, -1.037651097177562e-01, -6.082689056569043e-02, -2.888083868311618e-02, -1.653681708890641e-02, -6.385085449920612e-03, -1.219523527026374e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_gl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_gl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.262523733952668e-01, -1.260744951524712e-01, -1.144170795203890e-01, -1.142685728637905e-01, -7.004949687563339e-02, -7.008872995878497e-02, -3.394999981987709e-02, -9.715570433646487e-02, -2.026626486911265e-02, -5.196157573582608e-02, -8.290579162251946e-03, -8.333620739263201e-03, -1.540911453316271e-04, -1.859649446613552e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
