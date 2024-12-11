
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.520993093951740e+00, -1.038638909850309e+00, -2.407012152351580e-01, -1.379930824999966e-01, -5.452344745867088e-02, -3.824898431060312e-03, -7.923871875124550e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.081583906309191e+00, -2.083338682082220e+00, -1.466997079338614e+00, -1.467915151894629e+00, -3.122637415265397e-01, -3.126943299468267e-01, -1.838184372808560e-01, 7.539152113143308e-02, -7.016576369109429e-02, -5.466433811679944e-02, 1.368037809691398e-01, 2.413768612877522e-01, 7.612061504734248e-05, -1.610477432387999e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.925776125030286e-04, 1.420514145497902e-04, -1.915618167007073e-04, -7.866987417503018e-04, 1.115362495104119e-03, -7.817752887153918e-04, 1.947699480031190e-01, 4.180228328436082e-01, 1.943543253347252e-01, 4.676155955302708e-02, 7.035114068157315e+00, -6.147868400642953e+03, 1.101298687036864e+02, 2.641353630163130e+02, -5.315713541372420e+06, -6.408371565769667e+01, 8.575971354726938e-01, -5.499152802075459e+03, -3.602594773037514e+00, 4.666265396414997e-04, 2.333132698207498e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.380252567064424e-03, 9.364481524751313e-03, 1.412376655784023e-02, 1.409811870116033e-02, 5.047783474188717e-04, 6.022602808474899e-04, 1.210753098234890e-02, -4.209732007622868e-02, -3.712127698377490e-03, -4.695267997225012e-02, 9.494662670199024e-04, 8.016862846789767e-02, 4.392023553308677e-10, -6.324714924461792e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
