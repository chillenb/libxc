
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_1d_loos_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [5.215172681627910e-03, 7.322587152380774e-04, -7.631380191453105e-02, -1.390163373199700e-02, -9.831889298849955e-04, -1.224991473763316e-05, -6.885999589117415e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_1d_loos_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [7.274721715571542e-03, 7.274721715571542e-03, 7.933028563031505e-03, 7.933028563031505e-03, -1.016064353060499e-01, -1.016064353060499e-01, -2.620589233268663e-02, -2.620589233268663e-02, -1.954869999228176e-03, -1.954869999228176e-03, -2.449470832483967e-05, -2.449470832483967e-05, -1.377199440368949e-10, -1.377199440368949e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05