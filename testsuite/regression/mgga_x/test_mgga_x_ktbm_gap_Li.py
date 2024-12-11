
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_gap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.976398736438201e+00, -1.323254947836985e+00, -1.481788387162100e-01, -1.800802629910110e-01, -3.977279617549832e-02, -6.547810395808352e-03, -1.159490466946243e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_gap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.718145757254963e+00, -2.720703307070615e+00, -1.960781843815271e+00, -1.962069510705834e+00, -2.478675484708975e-01, -2.503880106938807e-01, -2.432473094534476e-01, -7.537828289184100e-03, -7.833991226093309e-02, -2.388575214404310e-04, -7.919794521042460e-03, -7.869311398791540e-03, -1.595427500235584e-04, -9.372483202730994e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.062426431782675e-04, 0.000000000000000e+00, -8.034311832874143e-04, -3.233389108314472e-03, 0.000000000000000e+00, -3.223793804888923e-03, -1.128842798565877e-01, 0.000000000000000e+00, -1.165481888985586e-01, -1.224924767711775e+01, 0.000000000000000e+00, -2.050575477996212e+01, -1.330468270152140e+02, 0.000000000000000e+00, -5.139511096602103e+04, -3.815824744803277e-01, 0.000000000000000e+00, -1.833378402036701e+01, -7.782701973454875e-01, 0.000000000000000e+00, -6.709827511996541e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.204557206646464e-02, 1.200858935682023e-02, 2.885670771623417e-02, 2.879083022691643e-02, 9.220595783013966e-03, 9.754044593218490e-03, 8.714567611964620e-02, 2.627805313310022e-04, 2.119157250687441e-01, 2.094036884604568e-05, 5.667385722495294e-06, 2.673383677240175e-04, 9.449449657431243e-11, 6.178972377385318e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
