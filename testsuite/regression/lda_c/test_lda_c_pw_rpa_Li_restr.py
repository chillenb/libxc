
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_rpa_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.152512939539556e-01, -1.048586507555029e-01, -6.730951779718612e-02, -4.852381159976971e-02, -3.181841321969579e-02, -1.441018519509033e-02, -1.028553268068961e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_rpa_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.248501712100399e-01, -1.142400172068727e-01, -7.532601791891039e-02, -5.532076203622987e-02, -3.702966543081664e-02, -1.721024670550128e-02, -1.269702998259678e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
