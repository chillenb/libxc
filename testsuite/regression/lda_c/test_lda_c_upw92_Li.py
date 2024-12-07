
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_upw92_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_upw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.360489405642389e-02, -8.385412276463074e-02, -4.975137714737145e-02, -1.811646790861236e-02, -1.091878802125036e-02, -6.762765143313467e-03, -1.679987314340586e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_upw92_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_upw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.027473190071718e-01, -1.025854238881466e-01, -9.266931328321852e-02, -9.253060553218832e-02, -5.681382078616274e-02, -5.685735150557309e-02, -2.108869150326046e-02, -1.251958401009215e-01, -1.306983641556995e-02, -7.220807045166354e-02, -8.503541054619921e-03, -8.599152628342575e-03, -1.980370534157488e-04, -2.887806913129876e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
