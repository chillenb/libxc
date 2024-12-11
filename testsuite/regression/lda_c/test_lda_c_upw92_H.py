
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_upw92_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_upw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.265521964901907e-02, -3.135883058125728e-02, -2.531275029020137e-02, -1.324446280574711e-02, -1.597212137658102e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_upw92_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_upw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.664839501881798e-02, -2.626256526841887e-01, -3.528744262943088e-02, -2.506012542581368e-01, -2.888994745979502e-02, -1.955766561047534e-01, -1.570879453978825e-02, -9.138595401040529e-02, -2.028897167193059e-03, -6.614317001770767e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
