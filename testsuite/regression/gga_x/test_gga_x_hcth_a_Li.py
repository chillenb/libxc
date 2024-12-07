
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hcth_a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.858850379993745e+00, -1.283578824922108e+00, -4.583074910332054e-01, -1.692774871395018e-01, -7.873364389810000e-02, -3.119533726723810e-01, -1.343098208280208e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hcth_a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.539698373470245e+00, -2.542094778038250e+00, -1.707483513591148e+00, -1.709087700073669e+00, -1.582762008509896e-01, -1.577232374674636e-01, -2.313655893160568e-01, -5.909932334774860e-02, -5.487572105604374e-02, -1.859683814300709e-02, -5.946704093932432e-02, -6.021910693868102e-02, -1.821353877562295e-02, -1.583711272888781e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hcth_a_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.068897842402278e-04, 0.000000000000000e+00, 1.067871475354576e-04, -1.700027928844150e-05, 0.000000000000000e+00, -1.616028416351442e-05, -2.194895664602235e-01, 0.000000000000000e+00, -2.196778761250797e-01, 2.785253447791583e+00, 0.000000000000000e+00, -3.385591715299086e+03, -1.096358435751806e+02, 0.000000000000000e+00, -1.218464854835985e+08, -2.945194789454017e+03, 0.000000000000000e+00, -2.949352006391051e+03, -3.617227731722009e+08, 0.000000000000000e+00, -1.077507404089028e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
