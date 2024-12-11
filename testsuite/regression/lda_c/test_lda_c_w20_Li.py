
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_w20_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_w20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.425419143977518e-02, -8.417155675405218e-02, -4.932983695058236e-02, -1.836780240671551e-02, -1.127120233783802e-02, -6.639535214256521e-03, -1.703579218938001e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_w20_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_w20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.037021647106557e-01, -1.034929455432983e-01, -9.331884264679284e-02, -9.314307785087628e-02, -5.634556269288526e-02, -5.639550993681925e-02, -2.122519392710994e-02, -1.113116574811539e-01, -1.348316303338271e-02, -6.803736193570492e-02, -8.486905088765545e-03, -8.588469125460853e-03, -2.011602201847377e-04, -2.897626854675126e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
