
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_lda0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.372220903614626e+00, -9.668078340131278e-01, -2.542349270400178e-01, -1.312405232080622e-01, -5.488329614580634e-02, -1.363035937194762e-02, -2.857169291148435e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_lda0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.812343640932097e+00, -1.813785827145332e+00, -1.274257931195360e+00, -1.275179398752086e+00, -3.319273797676667e-01, -3.318369357323119e-01, -1.727119478884188e-01, -1.041089405309125e-01, -7.204717986019669e-02, -5.399048549520417e-02, -1.782757095345346e-02, -1.781610254974271e-02, -3.787659299885614e-04, -3.815097492049078e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
