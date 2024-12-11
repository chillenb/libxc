
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_rpa_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.155250823232635e-01, -1.052125563799581e-01, -6.795434437966846e-02, -3.481939783409666e-02, -2.438979994734789e-02, -1.399545279676388e-02, -5.677529858586528e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_rpa_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.251567554158966e-01, -1.249520150925018e-01, -1.146040210043952e-01, -1.144320017138573e-01, -7.590622493612724e-02, -7.595501422387482e-02, -3.884377139828194e-02, -1.253910990376550e-01, -2.780924480990874e-02, -7.815262313510118e-02, -1.694173659565838e-02, -1.701803952987505e-02, -7.678507816236720e-04, -6.626708377385841e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
