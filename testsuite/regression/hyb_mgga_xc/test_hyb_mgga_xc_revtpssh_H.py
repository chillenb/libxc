
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_revtpssh_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.417960201343557e-01, -5.769906037876638e-01, -3.364565586120986e-01, -1.121021437462842e-01, -6.655625471888261e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_revtpssh_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.093519047583391e-01, -6.417726766666727e-02, -7.232215033532960e-01, -3.750660549421887e-02, -4.303459785086600e-01, -1.779664551223779e-02, -1.035459472900205e-01, -5.198220614025288e-04, -8.864394094821376e-03, -1.389137545715529e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.890445931862268e+00, 5.250111144181079e+00, 2.625055572090539e+00, -5.787904377181233e-02, 9.291475156410782e-02, 4.645737578205393e-02, -1.132921530309716e-01, 2.072897711459802e-01, 8.433340444525287e-03, -5.201161071927654e+00, 3.120596217694251e-01, 1.534716980246042e-01, -7.807850259261395e+00, 5.921000338488177e-03, 2.960521782301161e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.499403718557209e+00, -6.259673206761210e+00, 1.003140196448779e-01, -7.988552663510960e-02, 4.040594617326697e-02, -3.565829797203884e-02, 3.355372045775534e-04, -1.022365774873494e-03, -1.956338061102608e-09, -2.028226327276599e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
