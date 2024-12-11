
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rscan_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.424616477180734e-16, -7.326327785578107e-03, -2.285707917113025e-02, -1.593841857045590e-02, -2.073504165676227e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rscan_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.632783294297951e-17, -2.198763077908834e-01, 7.359563943424251e-03, -2.247469938001951e-01, 7.105801965022224e-03, -1.955456129516529e-01, -1.762854062536355e-02, -8.896050345208219e-02, -1.128159557960140e-10, -1.341559643559810e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.896102147959951e-17, 5.792204295919903e-17, 2.896102147959951e-17, 1.731945741128419e-02, 3.463891482256837e-02, 1.731945741128419e-02, 2.979455354468619e-01, 5.958910708937238e-01, 2.979455354468619e-01, 4.151700664717006e+01, 8.303401329434011e+01, 4.151700664717006e+01, 4.898845759228292e-01, 9.797691505075774e-01, 4.898845759228292e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.185660407366731e-32, -1.171760301055732e-32, -1.994694153238811e-02, -1.962178391414598e-02, -4.014056504853951e-02, -3.999471671072196e-02, -8.605250738555198e-03, -8.605027816335297e-03, -2.219986832594267e-10, -2.219986849902486e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
