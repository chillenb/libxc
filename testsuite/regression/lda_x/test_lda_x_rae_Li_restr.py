
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_rae_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rae", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.064413782234068e-01, -7.389962915150852e-02, -1.774173352333863e-02, -7.638187568725874e-03, -3.027623858355331e-03, -6.986082470433898e-04, -1.241868541024462e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_rae_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rae", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.419218376312091e-01, -9.853283886867803e-02, -2.365564469778483e-02, -1.018425009163450e-02, -4.036831811140442e-03, -9.314776627245197e-04, -1.655824721365950e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
