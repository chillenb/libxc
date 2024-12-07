
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.466366393455854e-01, -1.441616959223239e-01, 1.075038577121570e-01, -2.796630042101865e-02, 2.520492554661560e-02, -3.003782024866416e-02, -7.453872414620748e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.472354367728028e-01, -3.469282214868851e-01, -3.154676448648492e-01, -3.152653518780191e-01, 2.338616162836605e-01, 2.339263548836772e-01, -2.686300569880056e-02, -2.571791643821608e-01, 2.459132774365535e-02, 3.592172858381109e+00, -3.775061806447988e-02, -3.817429715840330e-02, -8.768713814015609e-04, -1.286685901891996e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.511726762709141e-04, -5.023453525418282e-04, -2.511726762709141e-04, -2.424683161729464e-04, -4.849366323458926e-04, -2.424683161729464e-04, 3.130811872663788e-02, 6.261623745327578e-02, 3.130811872663788e-02, -1.811611618777389e+01, -3.623223237554777e+01, -1.811611618777389e+01, 1.087334643244207e+02, 2.174669286488414e+02, 1.087334643244207e+02, 1.714784216102049e-03, 3.429568432281492e-03, 1.714784216102049e-03, 1.641371548530787e-05, 3.282748297850945e-05, 1.641371548530787e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.985446758652017e-02, 4.985446758652013e-02, 3.553663941110240e-02, 3.553663941110240e-02, -4.330573423252119e-02, -4.330573423252106e-02, 8.071395298799833e-01, 8.071395298798062e-01, -3.541908769592302e-01, -3.541908767150829e-01, -9.004560191500802e-08, -9.004560191627598e-08, -2.379411459816161e-19, -2.379573016813704e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
