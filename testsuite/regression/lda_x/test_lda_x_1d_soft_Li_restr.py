
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_1d_soft_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.921999316705462e-01, -4.766921111964913e-01, -9.886400396240110e-02, -1.394495503987517e-02, -1.283398976815608e-03, -2.384573812191876e-05, -2.586436010794137e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_1d_soft_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.000000000000000e-01, -4.999998786813096e-01, -1.679084426692110e-01, -2.549015483354474e-02, -2.417330814318519e-03, -4.585518789544902e-05, -5.069810674642363e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
