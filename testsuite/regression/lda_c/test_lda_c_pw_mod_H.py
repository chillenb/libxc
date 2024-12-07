
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_mod_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.247092647741959e-02, -3.118079983566954e-02, -2.518518753711948e-02, -1.327196219152886e-02, -1.569790621002125e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_mod_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.644712803456843e-02, -2.627091453975010e-01, -3.508811108098496e-02, -2.506001728069861e-01, -2.872211520383755e-02, -1.951180632676889e-01, -1.571598845292718e-02, -9.051731111310297e-02, -2.001663505771958e-03, -7.071279902173418e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
