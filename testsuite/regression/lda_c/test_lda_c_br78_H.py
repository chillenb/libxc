
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_br78_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.818644888472412e-18, 1.628752235452549e-18, -8.759047183282285e-19, -1.647626336568799e-19, -5.849627407620066e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_br78_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([8.602460060266283e-18, -9.546610762597553e-02, 1.042703589060507e-17, -8.772204870685375e-02, -1.127530811680329e-18, -5.633967930981414e-02, -2.174285577852613e-19, -1.674137611575252e-02, -7.325981654023164e-20, -8.208081079866615e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
