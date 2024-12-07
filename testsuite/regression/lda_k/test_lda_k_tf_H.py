
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_tf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_tf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.034006394076559e+00, 1.635289207408385e+00, 5.594379175393203e-01, 3.989891768981788e-02, 8.855168206558591e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_tf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_tf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.390010656794277e+00, 7.560583270560528e-10, 2.725482012347320e+00, 7.546554492421365e-10, 9.323965292322223e-01, 7.585283512412389e-10, 6.649819614977766e-02, 7.596254458785978e-10, 1.475861384993562e-04, 7.596333144487596e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
