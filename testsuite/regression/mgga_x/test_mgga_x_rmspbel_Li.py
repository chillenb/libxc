
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmspbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.861704186074203e+00, -1.239289626868440e+00, -2.643111879621778e-01, -1.701354939514651e-01, -5.788587100205460e-02, -2.054607074459901e-02, -3.450789298810724e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmspbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.588309493207475e+00, -2.590700421544033e+00, -1.817601973024349e+00, -1.819506895476759e+00, -3.572992445515785e-01, -3.574838551293583e-01, -2.307968438172038e-01, -2.608213483510491e-02, -8.150191024866647e-02, -8.296405943696783e-04, -2.750624612241474e-02, -2.722272306143004e-02, -5.541564195063563e-04, -1.983334294199956e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmspbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.244911138442756e-04, 0.000000000000000e+00, -4.227802707902367e-04, -2.027212724704902e-03, 0.000000000000000e+00, -2.023701007323087e-03, -2.467205510662198e-01, 0.000000000000000e+00, -2.472990483013464e-01, -4.751731407186125e+00, 0.000000000000000e+00, -4.925669667940220e-01, -1.252048786578518e+02, 0.000000000000000e+00, -3.158639222406673e+00, -2.103894862931148e-04, 0.000000000000000e+00, -4.673562674432135e-01, -1.441715745861419e-10, 0.000000000000000e+00, -2.393660790669037e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmspbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.528860571919767e-02, 1.526547811995671e-02, 2.428024148858703e-02, 2.432400531655886e-02, 8.825105540825775e-04, 9.492959081834605e-04, 1.104616813373910e-01, 5.036685166460381e-19, 3.617738711671697e-02, 6.841291692738774e-25, -6.152971376756863e-25, 1.981121746219198e-19, 8.014211170282593e-36, 1.087151430603148e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
