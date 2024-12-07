
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.132985658673928e+00, -8.785000605607791e-01, -2.566698641326014e-01, -9.749748231018845e-02, -5.651588604458772e-02, -2.167079231124399e-02, -4.056681300925707e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.156668071528647e+00, -1.157969691826493e+00, -8.888181309350734e-01, -8.890064524214405e-01, -2.812760010172698e-01, -2.863402738877457e-01, -1.237363094575487e-01, -2.737349701640090e-02, -5.535581089300004e-02, -8.767626108864656e-04, -2.899824439220382e-02, -2.855832944228061e-02, -5.856408303184373e-04, -4.163353859212650e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.863793601317622e-04, 0.000000000000000e+00, -2.853457697496460e-04, -1.192441858675314e-03, 0.000000000000000e+00, -1.188477157543912e-03, -9.967732695713413e-02, 0.000000000000000e+00, -9.959298165142694e-02, -4.340775960193646e+00, 0.000000000000000e+00, -7.839351174135163e-01, -8.719244544877296e+01, 0.000000000000000e+00, -5.031261013843191e+00, -3.128710755488361e-01, 0.000000000000000e+00, -7.437732719289286e-01, -1.366667006860442e+00, 0.000000000000000e+00, -5.242630082576518e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_sx_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.650902935625824e-02, -2.648259125362872e-02, -8.157024872584977e-03, -8.254199001172566e-03, 2.654143101500027e-02, 2.761474050974794e-02, 7.472340429003894e-02, -1.625832863081910e-05, 1.661811469548191e-01, -3.334000905035604e-09, -3.684614009310591e-08, -1.754669136917268e-05, -2.594053622491034e-19, -3.719614543218879e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
