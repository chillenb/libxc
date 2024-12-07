
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_blyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.735738702286960e-01, -3.272923722270201e-01, -1.411204877582666e-01, -7.899556259522919e-03, -9.568214428138680e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_blyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.546981718467329e-01, -2.361713832519181e-01, -4.692336173355593e-01, -2.499095375018285e-01, -2.077273566590535e-01, -1.966669686227087e-01, -1.447679917767642e-02, -3.555089138121423e-02, -1.913091761658668e-06, -2.453539843799460e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_blyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.667334931005368e-03, 2.924820117334720e-02, 2.192774932900085e-02, -1.067377311478225e-02, 4.646917650590301e-02, 3.480665078312204e-02, -3.951065090076041e-02, 3.986771514585075e-01, 2.989990296175629e-01, -7.204921915536025e-02, 1.702787037828197e+01, 1.277088117642037e+01, -3.577459915126658e-05, 7.157809872575426e-18, 5.368349375325281e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
