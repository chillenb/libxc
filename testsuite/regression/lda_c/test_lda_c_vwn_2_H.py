
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.240553225520312e-02, -3.111652546340102e-02, -2.512969361463137e-02, -1.327316222650603e-02, -1.560336981914811e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.637848240923836e-02, -2.667436302794922e-01, -3.502021717295453e-02, -2.545378034523861e-01, -2.865862581079150e-02, -1.985871396636988e-01, -1.570221632891319e-02, -9.270264806285215e-02, -1.993820721646598e-03, -3.637594515238935e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
