
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gapc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.916965457409173e-02, -4.503858931226228e-02, -1.053282901950010e-02, -1.575704962014495e-02, -3.521845875709775e-03, -2.173056662954462e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gapc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.131204041654947e-01, -1.129657615373115e-01, -9.435056440062231e-02, -9.423688302092102e-02, -2.969243152281138e-02, -2.970683140314061e-02, -2.329446862139587e-02, -1.107234596005314e-01, -1.090269121124049e-02, -3.920284321424716e-02, -1.043381063724564e-06, -1.046459144909086e-06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gapc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.533299601493494e-05, 9.066599202986987e-05, 4.533299601493494e-05, 1.286663843291485e-04, 2.573327686582970e-04, 1.286663843291485e-04, 4.504081737186971e-03, 9.008163474373942e-03, 4.504081737186971e-03, 2.608777866216978e+00, 5.217555732433957e+00, 2.608777866216978e+00, 1.523127877923529e+01, 3.046255755847057e+01, 1.523127877923529e+01, 2.938746196862215e-03, 5.877492393724430e-03, 2.938746196862215e-03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
