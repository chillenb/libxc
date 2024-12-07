
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_chachiyo_mod_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.202162294069258e-02, -8.208195232854490e-02, -4.810847327771447e-02, -1.846041714555220e-02, -1.129099089870542e-02, -6.523251375868468e-03, -1.329053320843528e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_chachiyo_mod_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.013571994436279e-01, -1.011996311800667e-01, -9.103891725435120e-02, -9.090696565328485e-02, -5.495206246420697e-02, -5.498926768251419e-02, -2.121391857674072e-02, -4.063877001419492e-01, -1.346848040711447e-02, -2.857592998439070e+00, -8.307458414556075e-03, -8.377650779214313e-03, -1.604196397718798e-04, -2.232252087805300e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
