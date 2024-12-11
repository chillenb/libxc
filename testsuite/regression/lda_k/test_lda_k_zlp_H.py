
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_zlp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([3.664579111013235e+00, 2.948518257268584e+00, 1.011667779863509e+00, 7.240009503962859e-02, 1.609225408547097e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_zlp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([6.098530759697489e+00, -9.101090629099744e-03, 4.907509722421230e+00, -6.687371665747512e-03, 1.684650266785102e+00, -1.462698282424010e-03, 1.206331179225828e-01, -3.370576520495683e-05, 2.681992977269465e-04, -3.559701985175083e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
