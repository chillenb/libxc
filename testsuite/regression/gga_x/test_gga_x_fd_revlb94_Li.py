
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_fd_revlb94_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.789871420816002e+00, -1.280042904843905e+00, -5.134543437589503e-01, -1.603446541496591e-01, -8.632174534324567e-02, -1.805045259341135e+00, -3.750525449751413e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_fd_revlb94_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.243295836570098e+00, -2.245443087735829e+00, -1.509515739306659e+00, -1.510910639364221e+00, -1.237500883626320e-01, -1.228177075522562e-01, -2.055207330725236e-01, 1.057344459971963e+00, -5.351998535700343e-02, 1.051046953591922e+00, 1.039312245187890e+00, 1.063579189393912e+00, 1.192425359101189e+00, 1.115194252258477e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_fd_revlb94_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.436537723233196e-04, 0.000000000000000e+00, -2.427838024326236e-04, -1.035430076064539e-03, 0.000000000000000e+00, -1.031939601891967e-03, -2.718605362377892e-01, 0.000000000000000e+00, -2.722286314886033e-01, -3.664600564547207e+00, 0.000000000000000e+00, -3.359482729590988e+04, -1.347404138749131e+02, 0.000000000000000e+00, -3.945638700580797e+09, -2.842138958154438e+04, 0.000000000000000e+00, -2.882166296981767e+04, -1.357422930430599e+10, 0.000000000000000e+00, -4.351785885112900e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
