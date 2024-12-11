
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc18_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.185032569822995e-02, -1.190097062742028e-01, -8.459522772962406e-03, -8.869171415249532e-07, -2.829459888978678e-11, -5.100201035653005e-07, -2.228660242877977e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc18_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([8.237598532466423e-04, 8.975921958345599e-04, -2.437778441694380e-02, -2.413400910525097e-02, -1.700618704169107e-02, -1.702786056094696e-02, -8.874817847060111e-07, -2.666704084723741e-03, -2.829507875457449e-11, -1.660775503392571e-04, -1.008993109451420e-06, -1.031332734917873e-06, -3.029388650563377e-12, -8.431670440306374e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
