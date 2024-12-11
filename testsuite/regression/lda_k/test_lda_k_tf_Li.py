
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_tf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_tf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.586610266226971e+01, 7.647751223760097e+00, 4.407994494310949e-01, 1.296212053322181e-01, 2.037695186516548e-02, 6.835104852651212e-04, 2.428649364816099e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_tf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_tf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.641965671658787e+01, 2.646728781131748e+01, 1.273539394947440e+01, 1.275708245150211e+01, 7.349780485918811e-01, 7.343530507361383e-01, 2.161068734465878e-01, 1.037631473699424e-03, 3.396159222779484e-02, 1.043718048703013e-06, 1.147407944868488e-03, 1.130778260108290e-03, 4.656519441405979e-07, 2.353363432516353e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
