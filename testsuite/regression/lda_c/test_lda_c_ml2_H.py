
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ml2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.203722610479388e-06, -2.204012369190679e-06, -2.193656627029464e-06, -2.192084915867751e-06, -2.201211301574628e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ml2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.254480844917342e-06, -2.153657918047929e+08, -2.212660964538969e-06, -1.552060144234801e+08, -2.208464219057577e-06, -3.136011526934679e+07, -2.191883847946710e-06, -5.985298405465421e+05, -2.198127010177126e-06, -6.308269922658465e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
