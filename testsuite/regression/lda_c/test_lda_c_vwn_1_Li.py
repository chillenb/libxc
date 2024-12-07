
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.400973374239162e-02, -8.420805652593509e-02, -4.968925199695142e-02, -1.805411625703760e-02, -1.097300011476463e-02, -6.795005810983662e-03, -1.645867534314511e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.032070021600402e-01, -1.029975604564644e-01, -9.310434334544836e-02, -9.292766020273713e-02, -5.683753847571509e-02, -5.688851278373848e-02, -2.097038942004052e-02, -1.124265673044852e-01, -1.310880013009209e-02, -6.761537399461726e-02, -8.544310996211398e-03, -8.642501424244574e-03, -1.969440941460110e-04, -2.758223675418979e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
