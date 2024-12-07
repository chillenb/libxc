
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_yukawa_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.993332294789877e-01, -4.372840868343616e-01, -2.188180071908034e-01, -2.466330981345907e-02, -5.955197813459447e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_yukawa_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.000351375999500e-01, -1.396154046167572e-11, -6.161481244402877e-01, -1.396383957377074e-11, -3.183538324190544e-01, -1.396321330839972e-11, -4.131034642924802e-02, -1.396263210769292e-11, -1.189489776047278e-05, -1.396262589533062e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
