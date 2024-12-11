
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
    ref_tgt = [-1.916072616222055e+00, -1.272475027880858e+00, -2.924865573350069e-01, -1.764873384460579e-01, -6.010158346463422e-02, -1.286436040416358e-02, -2.388872344511346e-04]
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
    ref_tgt = [-2.705768987027023e+00, -2.708409059175114e+00, -1.863217146609515e+00, -1.864951387221683e+00, -3.598623640686412e-01, -3.589216109058170e-01, -2.444557274395694e-01, -1.561340426140368e-02, -7.551276128699677e-02, -4.951970345515109e-04, -1.641894142156279e-02, -1.629911855640628e-02, -3.307631926202035e-04, -2.398543547209549e-04]
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
    ref_tgt = [-5.750259745696605e-04, 0.000000000000000e+00, -5.730699480854213e-04, -2.175458963100132e-03, 0.000000000000000e+00, -2.169519738186202e-03, -4.403426521233096e-02, 0.000000000000000e+00, -4.609310801027907e-02, -8.985981746598782e+00, 0.000000000000000e+00, -1.941890137658040e+01, -7.089629495226600e+01, 0.000000000000000e+00, -4.862660385930560e+04, -3.610344389196550e-01, 0.000000000000000e+00, -1.736302807638660e+01, -7.363452972307541e-01, 0.000000000000000e+00, 3.646365887093470e+04]
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
    ref_tgt = [2.170670443053563e-02, 2.169235871903156e-02, 2.448316930632547e-02, 2.449624009001935e-02, -5.540901358831672e-03, -5.715646978685622e-03, 2.579346918606421e-01, 2.482093954655395e-04, -3.870569709622535e-02, 1.981224060966626e-05, 5.361882384716857e-06, 2.524885528649801e-04, 8.940414062188752e-11, -1.349962017628787e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
