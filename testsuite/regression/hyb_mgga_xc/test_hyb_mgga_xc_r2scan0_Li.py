
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.553712753134264e+00, -1.084190630162943e+00, -2.593359585600693e-01, -1.382295523855105e-01, -5.388045976042749e-02, -5.520449675832904e-03, -1.660642897655507e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.037220005855198e+00, -2.039010756668838e+00, -1.408603451357251e+00, -1.409791124862318e+00, -2.412839197534362e-01, -2.790894399734227e-01, -1.838826623154518e-01, 7.342052413724240e-02, -7.465727982344160e-02, -9.138724344811605e-02, -9.148968588257865e-03, 2.406948429836222e-01, -4.594790983715226e-05, -1.347139219250735e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.480969810400665e-04, 3.376676096858215e-05, -1.475061752035829e-04, -5.692219619783855e-04, 1.486544711016959e-04, -5.664017721550684e-04, -1.759774983784816e-01, 3.686155290429969e-02, -1.244816668357080e-01, -5.349674397142201e-01, 4.249088087620090e+00, -6.149261413633222e+03, 7.422080438567718e+00, 1.567412608661205e+02, -5.315958140639327e+06, 1.831265699291777e+01, 5.982918252051843e+00, -5.496590141517169e+03, 3.100991507787625e+04, 1.294551542135600e+04, 6.472757710678000e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.274308498379820e-03, 7.266627942795725e-03, 9.518491949545269e-03, 9.498850837359020e-03, 4.111651168913503e-02, 2.878237822160230e-02, 2.307472410310615e-02, -1.849260579195225e-03, -4.815485447379697e-03, -1.852473212654703e-01, -9.235433548631923e-11, 8.018066139380518e-02, -7.436176314524375e-20, -1.666241956151310e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
