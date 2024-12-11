
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.880651588603539e+00, -1.259594304348813e+00, -2.935013929513394e-01, -1.725975363831420e-01, -6.055373102718194e-02, -1.313850966201449e-02, -2.428002535207951e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.627186164333787e+00, -2.629743235070332e+00, -1.812805856277286e+00, -1.814460747825570e+00, -3.636901483523439e-01, -3.628151092813912e-01, -2.373843865116973e-01, -1.626841118127996e-02, -7.650778811487145e-02, -5.159640547417598e-04, -1.710750556072199e-02, -1.698290704332911e-02, -3.446343575848293e-04, -2.390262821607970e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.203773437964934e-04, 0.000000000000000e+00, -5.185904739161867e-04, -2.036796247860552e-03, 0.000000000000000e+00, -2.030900248858776e-03, -6.850333581242717e-02, 0.000000000000000e+00, -7.101120948751077e-02, -8.026563050740341e+00, 0.000000000000000e+00, -1.158135219851722e+01, -8.693765555167826e+01, 0.000000000000000e+00, -2.894646011654660e+04, -2.149248983440642e-01, 0.000000000000000e+00, -1.035643173946059e+01, -4.383297625714894e-01, 0.000000000000000e+00, -2.406351843797153e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.723621766992505e-02, 1.722527855895929e-02, 1.961828694570545e-02, 1.962774340202268e-02, -5.086954260548769e-03, -5.246865529971769e-03, 2.043616919919143e-01, 1.481300915887135e-04, -3.542212591676425e-02, 1.179385707350514e-05, 3.191993968537218e-06, 1.507074009326763e-04, 5.322027027168175e-11, -1.230910302140307e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
