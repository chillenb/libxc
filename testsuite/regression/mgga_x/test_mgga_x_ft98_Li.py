
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ft98_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ft98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.862654917275302e+00, -1.359835431223187e+00, -4.043887988666439e-01, -1.729678381040903e-01, -8.020997593281863e-02, -6.286200614377475e-02, -9.623596837806285e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ft98_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ft98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.662487204531298e+00, -2.666178218400463e+00, -1.357981977680873e+00, -1.359433619219856e+00, -1.902073405655417e-01, -1.781019108433541e-01, -2.323647129073859e-01, 7.391646977304834e-01, -5.429507976551342e-02, 3.203195728791980e-01, -1.624374172136258e-02, 7.323487764717841e-01, -1.354209216464424e-04, 2.853793702152926e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ft98_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ft98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.945648538830897e-04, 0.000000000000000e+00, 2.959196340963372e-04, -1.392180764481128e-03, 0.000000000000000e+00, -1.387186064266592e-03, -2.682041959014217e-01, 0.000000000000000e+00, -2.785114425674420e-01, 8.688479801923631e+00, 0.000000000000000e+00, -2.011070313820309e+04, -1.524056634510954e+02, 0.000000000000000e+00, -9.706503910172896e+08, 2.706788244076469e+01, 0.000000000000000e+00, -1.735531017553982e+04, 5.897782935535014e+04, 0.000000000000000e+00, -9.035540882999723e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ft98_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ft98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-3.027155655994048e-06, -3.038551869650774e-06, 4.753269592607821e-03, 4.747461154632632e-03, 7.576924574439562e-03, 7.957627928973260e-03, 1.353440007136198e-02, 3.397887808904983e-02, 4.053647712507650e-02, 7.253308617054244e-02, -2.538670900466773e-18, 3.390826484343918e-02, -3.219755804565706e-35, 7.800718029108346e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ft98_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ft98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
