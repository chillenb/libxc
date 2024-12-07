
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_karasiev_mod_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.222333207986144e-02, -8.234683902468486e-02, -4.866874251758126e-02, -1.865328957897602e-02, -1.147229362871514e-02, -6.845891189368122e-03, -1.407872839960134e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_karasiev_mod_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.015066405407341e-01, -1.013485440073068e-01, -9.124146520706523e-02, -9.110885864424459e-02, -5.545250276715924e-02, -5.549039680537345e-02, -2.139642133192531e-02, -4.193438298653039e-01, -1.365928197561432e-02, -3.012412826646766e+00, -8.696040684415084e-03, -8.772566825943153e-03, -1.689114046953140e-04, -2.392560351657807e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
