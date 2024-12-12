
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.045468258004667e+00, -1.435533996768101e+00, -3.356557471896895e-01, -1.835335801922633e-01, -7.386817870343527e-02, -1.268642752128917e-02, -2.350306896239640e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.549539406547507e+00, -2.552030058665566e+00, -1.699921875130596e+00, -1.701391152806535e+00, -4.087655824102051e-01, -4.096106513866483e-01, -2.355348877636499e-01, -1.561340426140370e-02, -8.653874948830484e-02, -4.951970350154189e-04, -1.674679602829452e-02, -1.629911855640628e-02, -3.373913123213319e-04, -2.351424087719279e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.092935908308767e-04, 0.000000000000000e+00, -6.072157635758503e-04, -2.411291133908256e-03, 0.000000000000000e+00, -2.404586809401738e-03, -6.520053945425620e-02, 0.000000000000000e+00, -6.837457942330498e-02, -9.308568346307304e+00, 0.000000000000000e+00, -1.941890137658040e+01, -9.054524739678000e+01, 0.000000000000000e+00, -4.862660248404766e+04, 4.999863498803239e-02, 0.000000000000000e+00, -1.736302807638660e+01, 1.219610649816821e-01, 0.000000000000000e+00, -2.201401181436654e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.545950851090162e-02, 2.543927887397147e-02, 3.574340346096676e-02, 3.573627431928347e-02, 1.755731175167684e-02, 1.905214579371996e-02, 2.810235483774317e-01, 2.482093954655395e-04, 2.463836431560132e-01, 1.981223951831359e-05, -1.546874038373749e-08, 2.524885528649801e-04, -1.172569396682810e-16, 9.603217951389691e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
