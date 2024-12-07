
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.952703410905840e+00, -1.368982418171781e+00, -3.734515203493586e-01, -1.757249878011455e-01, -7.621995576972218e-02, -1.712967270431935e-02, -3.200241284136088e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.550715138372152e+00, -2.553153190062744e+00, -1.755941292512033e+00, -1.757927653805070e+00, -4.185900230364900e-01, -4.232015616763516e-01, -2.312409176649133e-01, -2.177864853405508e-02, -8.838541458411603e-02, -6.916765007599124e-04, -2.289804978503743e-02, -2.273321948180553e-02, -4.620011818408396e-04, -3.284408051171884e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.894044021665153e-05, 0.000000000000000e+00, -8.818299550735533e-05, -3.655140096492284e-04, 0.000000000000000e+00, -3.578683886885040e-04, -4.209471231281407e-02, 0.000000000000000e+00, -3.619269104773625e-02, -1.835381376137614e+00, 0.000000000000000e+00, -1.938331164652770e-01, -2.897488535600552e+01, 0.000000000000000e+00, -1.241225202889674e+00, -1.971703255551327e-01, 0.000000000000000e+00, -1.839282898563473e-01, -9.035686113784069e-01, 0.000000000000000e+00, -1.293364786529187e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.327870720652541e-05, 2.258918824304505e-11, 1.123693878720141e-04, 2.248648924103407e-17, 1.352398551014731e-03, 4.397761036150075e-12, 1.827085531199936e-02, 1.618681424715014e-24, 1.314703398403718e-07, 3.319461007794360e-22, 3.213323493052010e-13, 6.500920864628288e-25, 1.521512603732131e-27, 7.183304357563147e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
