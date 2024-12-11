
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_wigner_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.456408094633347e-02, -5.378894631516475e-02, -4.689202271551562e-02, -5.098436099893726e-05, -1.755792864805556e-08, -9.164025844426211e-03, -1.507874996767456e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_wigner_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.530692438600929e-02, -5.501207459315133e-02, -5.475954871044967e-02, -5.448497282359334e-02, -4.946961730794085e-02, -4.958929390112106e-02, -5.462288112509642e-06, -1.532465048910160e-01, -3.179549220893004e-09, -1.030577288537851e-01, -1.152389855584536e-02, -1.192529616352598e-02, -1.042657399329950e-04, -4.697752724993090e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
