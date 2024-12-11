
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_16_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.973794539509014e+00, -1.295975626018519e+00, -2.757086709350697e-01, -1.825511669779526e-01, -5.745072603690767e-02, -1.160778613645818e-02, -2.170573326850979e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_16_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.822974537717855e+00, -2.825738168346197e+00, -1.946499510410778e+00, -1.948312798818342e+00, -3.445531565287737e-01, -3.439227789049542e-01, -2.547674226755863e-01, -1.430258052306267e-02, -7.474210141020973e-02, -4.535958400263796e-04, -1.503961448451678e-02, -1.493078048833553e-02, -3.029759370146743e-04, -2.240302664141415e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_16_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.073879018743592e-04, 0.000000000000000e+00, -7.050150583843005e-04, -2.587788370219524e-03, 0.000000000000000e+00, -2.581156246745501e-03, -3.057099277857205e-02, 0.000000000000000e+00, -3.242120429414667e-02, -1.119050582351947e+01, 0.000000000000000e+00, -1.205681984195656e+01, -6.425366073528518e+01, 0.000000000000000e+00, -3.013144279591281e+04, -2.237238042610193e-01, 0.000000000000000e+00, -1.078168579221065e+01, -4.562735582305585e-01, 0.000000000000000e+00, 1.972767384630454e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_16_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.745095391681633e-02, 2.742892129052734e-02, 3.208708322088913e-02, 3.209867108384123e-02, -4.244472683132523e-03, -4.356270753576216e-03, 3.211882340626322e-01, 1.542908937342560e-04, -1.556434105852061e-02, 1.227667891333362e-05, 3.322711584175911e-06, 1.569814456267023e-04, 5.539893514072681e-11, -1.196740133845388e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
