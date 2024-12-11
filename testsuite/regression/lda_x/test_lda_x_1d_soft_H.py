
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_1d_soft_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.468531904803174e-01, -3.068666005322918e-01, -1.261575068109802e-01, -5.639491464231684e-03, -1.374579329888132e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_1d_soft_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.641082725254931e-01, 1.315750971247261e-17, -4.326585004941655e-01, -4.628571031392474e-17, -2.099713984388613e-01, -3.112651660365280e-18, -1.045994768134899e-02, 4.905299236604875e-19, -2.663521926040289e-06, -4.236481576491043e-22])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
