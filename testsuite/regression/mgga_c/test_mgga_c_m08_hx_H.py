
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_hx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.659855927172224e-01, -5.234823184384518e-02, 3.929270976867893e-03, -4.376324585498310e-02, -1.091651695863600e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_hx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.428999915139302e-01, -2.577184938049895e+03, -9.713201648424932e-02, -1.624153601474030e+03, -6.123298944440561e-02, 7.142605781460533e+02, 2.114936256125329e-02, 3.878969406011559e+00, -1.390937634987896e-02, -4.916786677155036e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.276372027289854e+00, -4.552744054579708e+00, -2.276372027289854e+00, -2.232788504724158e-02, -4.465577009448315e-02, -2.232788504724158e-02, 7.232257256603676e-02, 1.446451451320735e-01, 7.232257256603676e-02, 1.056264188964084e-01, 2.112528377928179e-01, 1.056264188964084e-01, -2.554663715473037e-02, -5.109327426078566e-02, -2.554663715473037e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.463929574546074e+00, 5.461176160248762e+00, 1.066756111179344e-01, 1.059607946413531e-01, 4.104461473324357e-02, 4.096213380888266e-02, -8.717820035533069e-02, -8.717767816017997e-02, -9.115098587145399e-06, -9.115098637145083e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
