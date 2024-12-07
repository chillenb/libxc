
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ob_pz_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.213027136135322e-02, -8.199544266782412e-02, -4.842868153365526e-02, -1.722554172918637e-02, -1.072249470585888e-02, -6.661620997943153e-03, -1.560171131644614e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ob_pz_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.016082712518856e-01, -1.014055742029235e-01, -9.121621355516288e-02, -9.104596317629489e-02, -5.433835196888073e-02, -5.439082258901853e-02, -1.983351122031625e-02, -1.185762910963332e-01, -1.272572098845675e-02, -7.070487005530250e-02, -8.439430036900227e-03, -8.531086176916398e-03, -1.963500975100758e-04, -2.349392619610799e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
