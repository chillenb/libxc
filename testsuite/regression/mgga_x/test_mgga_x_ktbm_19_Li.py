
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_19_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.985480226802128e+00, -1.309462363664720e+00, -2.316506095793083e-01, -1.828154022489181e-01, -5.144787690649345e-02, -9.043249993897381e-03, -1.695752065156638e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_19_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.808669038230168e+00, -2.811383301985986e+00, -1.967771622671213e+00, -1.969423742468412e+00, -3.134067548123136e-01, -3.138707272147367e-01, -2.526104664052400e-01, -1.119758296640967e-02, -7.600286566590020e-02, -3.550286247565628e-04, -1.177154204377430e-02, -1.168959718722206e-02, -2.371385446082387e-04, -1.769680997525226e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_19_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.195233343299470e-04, 0.000000000000000e+00, -8.167589042642024e-04, -2.986608628173402e-03, 0.000000000000000e+00, -2.979070092590343e-03, -2.723150760828431e-02, 0.000000000000000e+00, -2.902309216078833e-02, -1.294946187940332e+01, 0.000000000000000e+00, -8.053740161462819e+00, -6.407246759067792e+01, 0.000000000000000e+00, -2.008284246398127e+04, -1.491203960843269e-01, 0.000000000000000e+00, -7.202951398336303e+00, -3.041081762188085e-01, 0.000000000000000e+00, 2.114055932141738e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_19_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.320811201330434e-02, 2.317618159503687e-02, 3.256961049146134e-02, 3.255482562723316e-02, 8.211424056411619e-04, 9.357060203923675e-04, 2.497667463704400e-01, 1.034711624637820e-04, 6.200756837871549e-02, 8.182585612935390e-06, 2.214914819046578e-06, 1.053151248398809e-04, 3.692361485061679e-11, -3.934968879229380e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
