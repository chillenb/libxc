
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_hf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([6.636410858001109e-01, 2.420257406250247e-01, -2.586034173576066e-01, 5.145531758103584e-02, -2.282183369305660e-02, 3.642086316827027e-02, 6.843762812343532e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_hf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.614445465780875e+00, 1.609612545956096e+00, 3.279230483915631e-01, 3.296716025990853e-01, 1.805810788557888e-02, 3.118101182600802e-02, -5.976400268774534e-02, 4.548001728551482e-02, 4.939558281434521e-02, 1.479073684023458e-03, 4.897555378647542e-02, 4.740335283559552e-02, 9.879974086035655e-04, 7.023667532215377e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.675359756577834e-04, 0.000000000000000e+00, 2.668249746250431e-04, 7.967389910909186e-04, 0.000000000000000e+00, 7.946775072942144e-04, -1.119769569710089e-01, 0.000000000000000e+00, -1.098216084593511e-01, 4.078134845905737e+00, 0.000000000000000e+00, -1.335719169691490e+00, -2.703143023010941e+01, 0.000000000000000e+00, -8.568308511623091e+00, 4.514176022264509e-01, 0.000000000000000e+00, -1.267328187268999e+00, 2.305616945083401e+00, 0.000000000000000e+00, -8.928264370008897e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.261083444584565e-01, -1.251604472883519e-01, -2.290878049247460e-02, -2.312049511575982e-02, -2.420451385857275e-02, -2.756680682400031e-02, 3.413855882567327e+00, 1.488732769507488e-04, -5.649377130090578e-01, 3.052638182332111e-08, -2.930637710557795e-08, 1.606712254055283e-04, -3.504238293907074e-19, 3.405708666070041e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
