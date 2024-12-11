
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_lp96_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.529134996455095e-02, -3.256134985037752e-02, -1.466747921456116e-02, 6.619642233491228e-02, -2.336942317035114e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_lp96_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_lp96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.323169212015833e-02, -4.323169212015833e-02, -4.131516968254267e-02, -4.131516968254267e-02, -2.843899746130093e-02, -2.843899746130093e-02, 4.421223025803780e-02, 4.421223025803780e-02, -6.506592854883888e+00, -6.506592854883888e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
