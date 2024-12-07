
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_407_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.917232418412548e+00, -1.346420214686409e+00, -4.726205804916488e-01, -1.788226466368415e-01, -9.014784433818653e-02, -2.294206076918816e-03, -1.377259679726886e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_407_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.562180656731919e+00, -2.564506551712082e+00, -1.744747580342248e+00, -1.746249938012240e+00, -3.606516717983647e-01, -3.611717832084581e-01, -2.461041255651328e-01, 1.518046371334916e+00, -6.618800150630132e-02, 9.705818390828328e-01, -6.265158565091752e-03, -4.289449257244567e-03, -7.742865745989443e-04, 1.461232514206216e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_407_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.803856848738146e-05, 0.000000000000000e+00, 3.826580568426841e-05, -1.815065965277066e-04, 0.000000000000000e+00, -1.799867025714995e-04, -1.277607016125344e-01, 0.000000000000000e+00, -1.275588395527152e-01, 4.326336699379850e+00, 0.000000000000000e+00, 4.270225500603684e+02, -1.140413567261912e+02, 0.000000000000000e+00, 5.091167795669916e+04, 2.236984597164051e+00, 0.000000000000000e+00, 2.749790663160544e+00, -5.053886376710142e+00, 0.000000000000000e+00, 1.149556412589800e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
