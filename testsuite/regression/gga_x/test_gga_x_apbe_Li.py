
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_apbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.804605232339550e+00, -1.296669269743935e+00, -4.257241858856992e-01, -1.605832974356085e-01, -8.250774878001244e-02, -2.054641351535674e-02, -3.838587273222373e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_apbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.231461645685922e+00, -2.233599522528130e+00, -1.507664796223791e+00, -1.509028651065121e+00, -4.170082611338179e-01, -4.172184486872030e-01, -2.046453676699462e-01, -2.612247987395847e-02, -7.785016799746769e-02, -8.296438661267367e-04, -2.746514524145216e-02, -2.726742024976003e-02, -5.541556673707600e-04, -3.939542612080420e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_apbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.976061362758596e-04, 0.000000000000000e+00, -2.965889803738872e-04, -1.161959594256420e-03, 0.000000000000000e+00, -1.158267951468869e-03, -7.305081948989467e-02, 0.000000000000000e+00, -7.284731200752295e-02, -4.636264817313799e+00, 0.000000000000000e+00, -2.345635050464885e-01, -7.037396650323474e+01, 0.000000000000000e+00, -1.499839789851169e+00, -2.383770598820155e-01, 0.000000000000000e+00, -2.225975597150590e-01, -1.091827406385078e+00, 0.000000000000000e+00, -1.562838369132883e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
