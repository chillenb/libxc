
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_rpw92_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.264913690207143e-02, -3.135200352133619e-02, -2.530244497474775e-02, -1.323497975187583e-02, -1.602579090225354e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_rpw92_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.664456856379448e-02, -2.603291286234712e-01, -3.528290244799510e-02, -2.483188472216280e-01, -2.888150010318906e-02, -1.935815641895444e-01, -1.569682700051473e-02, -9.103300469425638e-02, -2.034751795432215e-03, -6.630626162079614e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
