
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_hl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_hl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.139507028997847e-02, -8.336652801778954e-02, -5.304473372219676e-02, -3.684226788065147e-02, -2.198623609467340e-02, -7.374589299367454e-03, -1.560932574071931e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_hl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_hl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.875705342044835e-02, -9.875705342044835e-02, -9.066973357910463e-02, -9.066973357910463e-02, -5.979882755779554e-02, -5.979882755779554e-02, -4.283214479215527e-02, -4.283214479215527e-02, -2.661277940219456e-02, -2.661277940219456e-02, -9.454753529287563e-03, -9.454753529287563e-03, -2.079323856774391e-04, -2.079323856774391e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
