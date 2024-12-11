
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_1d_exponential_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_exponential", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.767747049016671e-01, -2.472462849532396e-01, -1.094840913886592e-01, -5.308170372339496e-03, -1.339935881844782e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_1d_exponential_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_exponential", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.644090770132998e-01, -3.644090770132998e-01, -3.395104425317222e-01, -3.395104425317222e-01, -1.777022858121995e-01, -1.777022858121995e-01, -9.797334036618919e-03, -9.797334036618919e-03, -2.594234708929801e-06, -2.594234708929801e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
