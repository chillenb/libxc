
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ow_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.632737191926925e-18, 2.468356262339674e-18, -1.638678960517573e-18, -4.377653584527956e-19, -1.870152273617569e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ow_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.208227187354722e-17, -1.382002466278711e-01, 1.545919274050350e-17, -1.329418088016718e-01, -1.911266107178388e-18, -1.054026142289869e-01, -5.528376380804566e-19, -4.448092600636484e-02, -2.341497376201566e-19, -2.624160553124055e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
