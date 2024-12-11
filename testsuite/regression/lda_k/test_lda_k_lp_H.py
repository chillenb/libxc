
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_lp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.102994775096520e+00, 1.690754104297137e+00, 5.784126446220922e-01, 4.125218862538879e-02, 9.155513239892300e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_lp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_lp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.504991291827546e+00, 7.817022584788447e-10, 2.817923507161908e+00, 7.802514190302645e-10, 9.640210743701760e-01, 7.842554383685751e-10, 6.875364770906525e-02, 7.853900307885908e-10, 1.525918891133708e-04, 7.853981658285415e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
