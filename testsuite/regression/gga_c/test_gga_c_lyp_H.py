
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_lyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.912425880239539e-15, -1.162876981227630e-14, -4.572593003903493e-14, -4.340497113512425e-13, -2.865040369647735e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_lyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.312416569490048e-15, -2.361713832519181e-01, 1.092732512834101e-15, -2.499095375018283e-01, 1.030338509195684e-14, -1.966669686227086e-01, -5.740279716350014e-13, -3.555089138121423e-02, -9.431026302128840e-11, -2.453539843799460e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_lyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.511300796859309e-16, 2.924820117334720e-02, 2.192774932900085e-02, -1.600436048283018e-15, 4.646917650590301e-02, 3.480665078312204e-02, -5.496715940311902e-14, 3.986771514585075e-01, 2.989990296175629e-01, 3.436563772335950e-11, 1.702787037828197e+01, 1.277088117642037e+01, 1.544996977497337e-23, 7.157809872575426e-18, 5.368349375325281e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
