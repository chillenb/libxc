
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_erf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.687960884272062e-01, -4.064230349253850e-01, -1.877116809867429e-01, -1.168630273748266e-02, -1.493915762998472e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_erf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.705756348565223e-01, -3.479388926669797e-14, -5.861810779435571e-01, -3.473565846893242e-14, -2.863975513285946e-01, -3.490916890880364e-14, -2.194498257431466e-02, -3.490555950835343e-14, -2.987341820536393e-06, -3.490658540625718e-14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
