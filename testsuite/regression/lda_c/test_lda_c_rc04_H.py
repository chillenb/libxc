
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_rc04_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.644959847326045e-02, -2.536977880621551e-02, -1.980920815426082e-02, -8.075574116645294e-03, -4.650589910470743e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_rc04_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.970444624885109e-02, -1.640478516562395e+03, -2.871091227888149e-02, -1.411360634519885e+03, -2.329870207722266e-02, -6.442522001096158e+02, -1.025259735837554e-02, -7.013531148739843e+01, -6.183567543199946e-04, -1.899779106974485e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
