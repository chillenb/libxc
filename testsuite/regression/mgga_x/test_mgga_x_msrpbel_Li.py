
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_msrpbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.893861794429124e+00, -1.336002730665317e+00, -4.133368527269097e-01, -1.700052304890300e-01, -7.935492430695913e-02, -2.055687026651119e-02, -3.838588870220200e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_msrpbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.443127515090282e+00, -2.445425539724172e+00, -1.669970364305389e+00, -1.671641627055910e+00, -2.604834107649670e-01, -3.585007985188661e-01, -2.223630661277934e-01, -2.615912580860030e-02, -7.739192777572948e-02, -8.296468243503926e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -3.939545845223386e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msrpbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.387401697712091e-04, 0.000000000000000e+00, -1.381264977893539e-04, -5.845726014390431e-04, 0.000000000000000e+00, -5.799379079595347e-04, -2.221794782384410e-01, 0.000000000000000e+00, -9.376690422980812e-02, -2.200204485000121e+00, 0.000000000000000e+00, -3.092674806204462e-192, -6.217805653545837e+01, 0.000000000000000e+00, 0.000000000000000e+00, -5.342829532086223e-172, 0.000000000000000e+00, -6.754522588623127e-181, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msrpbel_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msrpbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.297484197209639e-06, 6.104095812133021e-12, 4.673825236508773e-05, 9.332414857668745e-18, 3.059846212015307e-02, 9.508138195974442e-11, 4.110067028894033e-03, 0.000000000000000e+00, 6.656553657614075e-07, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
