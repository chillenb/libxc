
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_1d_loos_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.998137371611306e-02, -7.972354265942255e-02, -6.683959095913442e-02, -2.644322831552025e-03, -2.858200132028481e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_1d_loos_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.531687847361092e-02, -3.531687847361092e-02, -5.563013569525783e-02, -5.563013569525783e-02, -9.727253620981545e-02, -9.727253620981545e-02, -5.216553242379718e-03, -5.216553242379718e-03, -5.716264210897054e-07, -5.716264210897054e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
