
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_teter93_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.831087975699697e+00, -1.289838650924445e+00, -3.386583440439189e-01, -1.750760804052503e-01, -7.317217333686987e-02, -1.817683170143001e-02, -3.674397357485304e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_teter93_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.418431924486606e+00, -2.420375149675929e+00, -1.700448880255521e+00, -1.701693815778451e+00, -4.420942127991806e-01, -4.419701671411548e-01, -2.303856441783531e-01, -1.378598789919417e-01, -9.608278092162034e-02, -7.212236460293860e-02, -2.376585589619473e-02, -2.375017066577959e-02, -4.897167681800158e-04, -4.886408120727734e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
