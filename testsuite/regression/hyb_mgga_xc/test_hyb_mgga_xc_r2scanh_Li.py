
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scanh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.859324290703592e+00, -1.296151047523367e+00, -3.082396244635167e-01, -1.658350701253290e-01, -6.465654401988116e-02, -6.411011910385498e-03, -1.876993136116163e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scanh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.437697476315079e+00, -2.439860937386196e+00, -1.682760228865714e+00, -1.684198513685538e+00, -2.796768260027686e-01, -3.250394556611073e-01, -2.203381184796071e-01, 1.199326128508919e-01, -8.690263846773223e-02, -9.087760437759503e-02, -1.057705277473448e-02, 2.892384327634254e-01, -5.298704767421860e-05, -1.347139219250735e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scanh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.810930533449380e-04, 3.376676096858215e-05, -1.803840863411577e-04, -6.979318014842321e-04, 1.486544711016959e-04, -6.945475736962516e-04, -2.148591533446079e-01, 3.686155290429969e-02, -1.530641554932795e-01, -1.066869736419073e+00, 4.249088087620090e+00, -7.379538605168628e+03, -6.767629560330803e+00, 1.567412608661205e+02, -6.379165442893279e+06, 2.137689656629614e+01, 5.982918252051843e+00, -6.596506461645808e+03, 3.591734655131590e+04, 1.294551542135600e+04, 6.472757710678000e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scanh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.042843426521164e-03, 9.033626759820248e-03, 1.187261632222993e-02, 1.184904698760643e-02, 5.089900733955078e-02, 3.609804717851150e-02, 4.382598928409215e-02, 1.391720766532693e-02, 3.170569714841961e-02, -1.848125058635729e-01, -7.604581440557932e-11, 9.621679370734562e-02, -5.590927665126633e-20, -1.666241956151310e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
