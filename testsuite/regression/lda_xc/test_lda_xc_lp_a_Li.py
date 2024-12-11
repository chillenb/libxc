
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_lp_a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.027726033610240e+00, -1.407800554977624e+00, -3.379834661787608e-01, -1.935186330097218e-04, -3.930556726102178e-08, -1.330703261947269e-02, -1.840154599238317e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_lp_a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_lp_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.709120759059511e+00, -2.698163466448449e+00, -1.880665184745820e+00, -1.873478799579714e+00, -4.502136004787691e-01, -4.510761930359063e-01, -6.457059599169080e-05, -5.817134626183228e-01, -1.310186244760710e-08, -2.307073163067428e-01, -1.745446742761925e-02, -1.803733473391706e-02, -1.274528289614216e-04, -5.735070767334452e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
