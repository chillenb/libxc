
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mk00b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.740593680813345e-01, -1.478501924989560e+00, -1.996802599619640e-01, -2.059008381162861e-01, -4.527301061901178e-02, -1.190360698124814e-01, -5.337789217540026e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mk00b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.056779868234697e-01, -4.068941463454494e-01, -4.275539422380495e+00, -4.286391129492229e+00, -2.743295013279453e-01, -2.853784731222090e-01, -6.115293380479158e-01, -1.703698538972342e-02, -9.600138110120610e-02, -7.196928965816631e-03, -1.678670338132986e-02, -1.719070638463848e-02, -7.113446318604250e-03, -6.207317596699002e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.203442227636064e-04, 0.000000000000000e+00, -1.199259913458914e-04, -4.887381788060860e-04, 0.000000000000000e+00, -4.871353574556581e-04, -8.551479092689482e-02, 0.000000000000000e+00, -8.554884904510074e-02, -1.854383167779579e+00, 0.000000000000000e+00, -1.347690283898541e+03, -4.977717225998308e+01, 0.000000000000000e+00, -4.851170961207286e+07, -1.172313419604260e+03, 0.000000000000000e+00, -1.174000906457912e+03, -1.440146351758661e+08, 0.000000000000000e+00, -4.289928860780579e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00b_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-8.770832143589520e-05, -8.794654995949864e-05, -2.532542615386366e-02, -2.538789394082640e-02, -1.109577075968555e-02, -1.185684135084096e-02, -2.310943857110715e-01, -8.418105649152133e-06, -1.088925804043181e-01, -6.924487218530829e-11, -2.051798140241902e-10, -5.624287865557094e-06, -1.114397214380004e-21, -6.055740083204672e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00b_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([7.016665714871616e-04, 7.035723996759891e-04, 2.026034092309093e-01, 2.031031515266112e-01, 8.876616607748439e-02, 9.485473080672768e-02, 1.848755085688572e+00, 6.734484519321706e-05, 8.711406432345451e-01, 5.539589774824663e-10, 1.641438512193521e-09, 4.499430292445676e-05, 8.915177715040032e-21, 4.844592066563738e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
