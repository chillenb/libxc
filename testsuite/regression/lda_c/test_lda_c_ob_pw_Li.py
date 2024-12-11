
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ob_pw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.211489976842117e-02, -8.217072159118168e-02, -4.801566537808586e-02, -1.745480180778223e-02, -1.101529767345388e-02, -6.943687186544232e-03, -8.463603134818294e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ob_pw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.014441676632140e-01, -1.012822720487081e-01, -9.114848569308075e-02, -9.100977754442582e-02, -5.486443998644737e-02, -5.490797078896569e-02, -2.018072881500206e-02, -1.221805328755962e-01, -1.290091896582260e-02, -7.722025982784557e-02, -8.986357488901300e-03, -9.081969064813679e-03, -9.310990706266003e-05, -1.730475763839020e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
