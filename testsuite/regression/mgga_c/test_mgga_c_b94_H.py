
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b94_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.524470345550301e-13, -3.124844485612587e-12, -1.739531790882429e-12, -1.263319423043715e-14, -8.679769260059780e-14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b94_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.636048268804835e-04, -2.108708736554713e-02, -5.848029010874804e-03, -1.592161267653534e-02, -4.871963164172077e-03, -7.053206556971921e-03, -3.353633255141811e-03, -1.034741892589128e-03, -1.476825890515469e-05, -7.296755975194636e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.691959742945360e-03, 0.000000000000000e+00, 0.000000000000000e+00, 7.243601437147669e-03, 0.000000000000000e+00, 0.000000000000000e+00, 2.837345058575989e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.006626141961861e+00, 0.000000000000000e+00, 0.000000000000000e+00, 3.146931750855628e+01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [2.792602831735302e-16, 0.000000000000000e+00, 9.375235172500671e-14, 0.000000000000000e+00, 4.341864993693810e-13, 0.000000000000000e+00, 2.744773551351583e-15, 0.000000000000000e+00, 1.702194282269896e-14, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.596022095904731e-02, 0.000000000000000e+00, -1.245384657435241e-02, 0.000000000000000e+00, -9.761062914329547e-03, 0.000000000000000e+00, -6.595792529865997e-03, 0.000000000000000e+00, -2.155954860488591e-05, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
