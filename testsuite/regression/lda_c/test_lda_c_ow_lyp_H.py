
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ow_lyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.461559222413806e-18, 2.307866178362456e-18, -1.532133674491526e-18, -4.093022876180321e-19, -1.748556821606693e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ow_lyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.129669450078046e-17, -1.292146032159449e-01, 1.445405131136811e-17, -1.242980638187495e-01, -1.786997474354242e-18, -9.854944045212119e-02, -5.168926813839707e-19, -4.158882017097002e-02, -2.189255531589221e-19, -2.453540228187092e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
