
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_chachiyo_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.211960591364288e-02, -3.086576356679071e-02, -2.509093332129126e-02, -1.360858521118982e-02, -1.392569758108454e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_chachiyo_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.599076370957663e-02, -2.146897624715849e-01, -3.465656456150395e-02, -2.053471901458296e-01, -2.847719128276222e-02, -1.625472944495029e-01, -1.603021381360960e-02, -8.030921207906742e-02, -1.820551299029395e-03, -6.534877253410159e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
