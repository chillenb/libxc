
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_karasiev_mod_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.224425058116456e-02, -3.099703405339521e-02, -2.525202613507393e-02, -1.379673166558105e-02, -1.435703851876876e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_karasiev_mod_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.609504140070233e-02, -1.735256870013115e+03, -3.476774024825324e-02, -1.488449878429980e+03, -2.862196925882231e-02, -6.882827178224911e+02, -1.622240748937450e-02, -9.037616380357865e+01, -1.875619283109221e-03, -3.321699481048612e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
