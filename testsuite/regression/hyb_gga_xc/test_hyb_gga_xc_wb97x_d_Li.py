
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.376602684603128e+00, -9.620385436050141e-01, -2.699661680564543e-01, -6.844587795908660e-02, -1.555819268705663e-02, 8.495972768797839e-03, 1.259634189821764e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.786746399209519e+00, -1.788303695117748e+00, -1.180929457597464e+00, -1.181896227009509e+00, -2.879849511185851e-02, -2.756468147087919e-02, -1.137569767157881e-01, 4.253753036449013e-01, -1.291462964313682e-02, 2.619653795807802e-01, 1.019270427032171e-02, 1.066294893841030e-02, 2.778762705585053e-05, 5.611358098517512e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.029296172542336e-04, 0.000000000000000e+00, -1.024402232090240e-04, -6.200736579773938e-04, 0.000000000000000e+00, -6.177995556876565e-04, -1.781017600017058e-01, 0.000000000000000e+00, -1.787374228751928e-01, 4.285741608578593e+00, 0.000000000000000e+00, 8.149677043829978e+01, -2.632828220382603e+01, 0.000000000000000e+00, 9.647816409188030e+03, 8.916999680690219e-01, 0.000000000000000e+00, 9.591307765256576e-01, 1.510274926145140e+00, 0.000000000000000e+00, 2.530573044773825e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
