
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revtpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.877176948974964e-11, -2.005414251587169e-11, -6.389695312783399e-12, -6.346648725874452e-14, -1.622495865002030e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revtpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.417727842988234e-02, -6.417726766666716e-02, -3.750676844196811e-02, -3.750660549421859e-02, -1.779670976779555e-02, -1.779664551223781e-02, -5.198223461042443e-04, -5.198220614024774e-04, -1.389340921608548e-09, -1.389137544941468e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.625055572090539e+00, 5.250111144181079e+00, 2.625055572090539e+00, 4.645737578205393e-02, 9.291475156410782e-02, 4.645737578205393e-02, 1.036448855727892e-01, 2.072897711459802e-01, 8.433340444525287e-03, 1.560298108856144e-01, 3.120596217694251e-01, 1.534716980246042e-01, 2.960500169450684e-03, 5.921000338488177e-03, 2.960521782301161e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.260717114935597e+00, -6.259673206761210e+00, -7.987366986089077e-02, -7.988552663510960e-02, -3.565601741328540e-02, -3.565829797203884e-02, -1.022365919131974e-03, -1.022365774873494e-03, -2.028226327034058e-09, -2.028226327276599e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
