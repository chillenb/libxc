
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_b97_3c_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.850969625576684e-01, -6.024041764347213e-01, -3.608517683205325e-01, -1.989363770981514e-01, -1.724978154968662e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_b97_3c_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.145741238160742e-01, -1.606256471958725e-01, -7.987428687523451e-01, -1.925736696730016e-01, -4.416413406262741e-01, -1.940994355802016e-01, -5.983337484331671e-02, 1.195687320469607e-01, -2.287254965564517e-02, 6.425936286484460e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_b97_3c_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.900160544226089e-02, 0.000000000000000e+00, -9.130294641595297e+20, -1.120094363004059e-03, 0.000000000000000e+00, -6.823023874413133e+20, -8.071461282778085e-02, 0.000000000000000e+00, -2.847431545939925e+20, -2.296161111387459e+01, 0.000000000000000e+00, 1.252765100318053e+20, -4.781311546732048e+01, 0.000000000000000e+00, 1.592598723198910e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
