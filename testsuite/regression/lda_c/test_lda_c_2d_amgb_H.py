
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_2d_amgb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.513346767568373e-02, -2.387848589680996e-02, -1.812672912312458e-02, -6.344313490608892e-03, -1.191096596891127e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_2d_amgb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.900132136972806e-02, -5.575974682279217e-01, -2.767123094205718e-02, -5.198473821896971e-01, -2.157838410417236e-02, -3.311698788447083e-01, -8.463803078108031e-03, -6.797854028303658e-02, -1.708672533223165e-04, -7.591355268090504e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
