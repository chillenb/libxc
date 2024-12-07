
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_karasiev_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.224425049389445e-02, -3.099703394966063e-02, -2.525202589500547e-02, -1.379673001066356e-02, -1.435645919868122e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_karasiev_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.609504135977162e-02, -2.169145428020768e-01, -3.476774019938525e-02, -2.076876393165800e-01, -2.862196914183453e-02, -1.653960924158233e-01, -1.622240659217261e-02, -8.346899186889002e-02, -1.875581147452029e-03, -7.106104033612514e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
