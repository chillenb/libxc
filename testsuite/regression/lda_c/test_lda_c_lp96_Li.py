
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_lp96_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.295137982794314e-02, -4.977624354169614e-02, -1.908885092375633e-02, 2.481691161817315e-02, 8.287814848097476e-02, -1.152564029255434e+00, -6.982280723295282e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_lp96_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.536894909416093e-02, -5.536894909416093e-02, -5.321783543911868e-02, -5.321783543911868e-02, -3.167514723941883e-02, -3.167514723941883e-02, 2.653208362522020e-03, 2.653208362522020e-03, 7.466787580584519e-02, 7.466787580584519e-02, -4.629995266527123e-02, -4.629995266527123e-02, -2.306197913316270e+03, -2.306197913316270e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
