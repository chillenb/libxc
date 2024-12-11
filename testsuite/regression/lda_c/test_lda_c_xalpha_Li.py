
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_xalpha_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_xalpha", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.680722110165282e-01, -6.026811446987398e-01, -1.446909204968727e-01, -7.845059221549272e-02, -3.110930729507142e-02, -5.697580450806870e-03, -1.063910440748393e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_xalpha_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_xalpha", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.156907708062514e+00, -1.157950113097517e+00, -8.032325905714398e-01, -8.039162561514385e-01, -1.929622318414474e-01, -1.928801704581723e-01, -1.046331787982501e-01, -7.250312031208819e-03, -4.147908342103639e-02, -2.299464591143261e-04, -7.624196059083783e-03, -7.568744669732066e-03, -1.535910253655485e-04, -1.091891863975440e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
