
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc18_1_H_restr_1_zk():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.324605661640256e-02, -3.094553058293975e-02, -6.027211539616216e-03, -1.137772084106625e-04, -1.189405286136638e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc18_1_H_restr_1_vrho():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.734345184082309e-02, -6.263297330663986e-02, -1.210654799902035e-02, -2.275750885117340e-04, -2.378810594908231e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
