
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_rel_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.215285350666475e-01, -5.573077827079971e-01, -3.259932449037132e-01, -8.706227855175182e-02, -4.101560911074719e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_rel_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.286418929883586e-01, 6.282043384087110e-05, -7.430317551512121e-01, 4.528845945406257e-05, -4.346485965742627e-01, 9.063297365108358e-06, -1.160828654335567e-01, 1.726355874642760e-07, -5.468747927240288e-03, 1.805217607342424e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
