
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_2d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.214697819138757e-01, -6.974663908192038e-01, -3.119904359563090e-01, -4.305743883062543e-02, -4.402759543148431e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_2d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.232204672870818e+00, -7.111293356301837e-08, -1.046199586228811e+00, -7.101395650520039e-08, -4.679856539344745e-01, -7.128707544743884e-08, -6.458615824601700e-02, -7.136441003199322e-08, -6.604139391831979e-04, -7.136496481315316e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
