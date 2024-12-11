
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_xalpha_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_xalpha", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.108113892881217e-01, -2.786878614251052e-01, -1.630034201800214e-01, -4.353126875291567e-02, -2.050780469072438e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_xalpha_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_xalpha", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.144151857174970e-01, -4.015595787104172e-17, -3.715838152334754e-01, 9.169596101004977e-18, -2.173378935733669e-01, 5.076635854447888e-19, -5.804169167062509e-02, -2.526273250668095e-18, -2.734373990693003e-03, 2.583472648149202e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
