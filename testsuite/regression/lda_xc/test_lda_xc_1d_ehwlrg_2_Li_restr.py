
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_1d_ehwlrg_2_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.636363950957089e+02, -1.206088594993066e+01, -1.282514293435651e-01, -2.929454923706422e-02, -5.499971137962186e-03, -3.858984574775544e-04, -2.601285755709344e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_1d_ehwlrg_2_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.847461273612228e+02, -4.705302375546793e+01, -1.987295245452780e-01, -4.685936956775506e-02, -8.820442971608894e-03, -6.189798234628641e-04, -4.172462352108474e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
