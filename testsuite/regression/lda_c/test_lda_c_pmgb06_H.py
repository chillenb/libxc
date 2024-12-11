
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pmgb06_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.389273615932663e-03, -5.743597129362412e-03, -7.826815346425060e-03, -1.195162758664319e-02, -1.569796203642425e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pmgb06_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.333127637492397e-03, -7.732916871163478e+01, -4.633409944365291e-03, -7.187659068916899e+01, -6.298710601310054e-03, -4.081149450707054e+01, -1.289666449331382e-02, -4.251051062322159e-01, -2.001664170773116e-03, -7.028459980560899e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
