
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_chachiyo_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.202160791034865e-02, -8.208194043831288e-02, -4.810847160153250e-02, -1.829333246630166e-02, -1.129027143311858e-02, -6.523197111752793e-03, -1.310257135569557e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_chachiyo_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.013794647094622e-01, -1.011773927961212e-01, -9.105756201912912e-02, -9.088834194603107e-02, -5.494680974755781e-02, -5.499452323163227e-02, -2.111516845338238e-02, -1.065609057456227e-01, -1.346806831733353e-02, -6.441642231860754e-02, -8.297584270796099e-03, -8.387600616938985e-03, -1.538644889267251e-04, -2.319909169389317e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
