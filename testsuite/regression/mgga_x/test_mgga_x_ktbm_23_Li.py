
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_23_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.997718466738898e+00, -1.312352124723686e+00, -2.380165141594917e-01, -1.843313580829335e-01, -5.220720389725367e-02, -9.490421145255758e-03, -1.777104603328339e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_23_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.842622147827178e+00, -2.845377896956543e+00, -1.984386635230294e+00, -1.986098036448706e+00, -3.168196556016834e-01, -3.170948745483035e-01, -2.558096005023315e-01, -1.178263229489187e-02, -7.546985963157013e-02, -3.736015962143460e-04, -1.238734280238501e-02, -1.230030534707306e-02, -2.495442530747052e-04, -1.844471470301399e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.149513893703253e-04, 0.000000000000000e+00, -8.122102009978583e-04, -2.967853375769318e-03, 0.000000000000000e+00, -2.960354706427419e-03, -3.164898663607939e-02, 0.000000000000000e+00, -3.353581277887437e-02, -1.288973124395948e+01, 0.000000000000000e+00, -7.633648903577449e+00, -6.718043068825099e+01, 0.000000000000000e+00, -1.902429481903674e+04, -1.412620725929324e-01, 0.000000000000000e+00, -6.827480347606440e+00, -2.880784940111475e-01, 0.000000000000000e+00, 1.718756491559960e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.571877762442013e-02, 2.568757102514922e-02, 3.443740929518303e-02, 3.442856477735100e-02, -1.102795760131855e-04, -3.521870141369821e-05, 2.832624205213301e-01, 9.805023993908597e-05, 4.906762905815983e-02, 7.751284492058696e-06, 2.098182003320237e-06, 9.979964321730089e-05, 3.497735408449008e-11, -5.585325428749501e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
