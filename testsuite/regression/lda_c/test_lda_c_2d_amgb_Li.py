
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_2d_amgb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.632466364725027e-01, -1.515432006248879e-01, -7.776705222524211e-02, -1.091771590719664e-02, -4.454444891955764e-03, -1.519761201311801e-03, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_2d_amgb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.728311393929760e-01, -1.722796727398880e-01, -1.639205391961910e-01, -1.634339613653523e-01, -9.740718431037636e-02, -9.753365879328603e-02, -1.388376201599880e-02, -1.425177020222258e-01, -6.092015091131279e-03, -4.309049257157415e-02, -2.238072243278154e-03, -2.271251385401924e-03, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
