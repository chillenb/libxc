
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_1d_ehwlrg_1_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.566081304272651e+02, -1.562868662259504e+01, -1.256983511728179e-01, -2.649838603848974e-02, -4.528981521071394e-03, -2.736312763437722e-04, -1.222826368484808e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_1d_ehwlrg_1_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.343782960830568e+03, -6.185884203468991e+01, -1.982518557445550e-01, -4.327455149177588e-02, -7.417089251971121e-03, -4.482070044432125e-04, -2.002989591552354e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
