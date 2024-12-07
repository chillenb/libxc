
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_ksdt_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.827032743627954e+00, -1.286785214073452e+00, -3.387307289153397e-01, -1.580971975070978e-01, -6.910778474339237e-02, -1.822971301294098e-02, -3.749160167126807e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_ksdt_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.414727896945860e+00, -1.696914903916519e+00, -4.419470082092227e-01, -2.053367680609880e-01, -8.974004653586264e-02, -2.385299464758158e-02, -4.986354419044105e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
