
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tfvw_opt_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.358487990016650e+01, 7.786312074534663e+00, 2.233481682340969e+00, 1.029768627623730e-01, 5.712668624393473e-02, 1.848075367657922e+00, 8.127816938124014e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tfvw_opt_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.588826662946192e+01, 1.592152094246347e+01, 6.421390308255878e+00, 6.435208457486271e+00, -1.411095262212045e+00, -1.417273557804175e+00, 1.386006527913237e-01, -1.830459080802733e+00, -1.925260481852211e-02, -7.249254865284092e-01, -1.815457113928690e+00, -1.878846842345811e+00, -8.496494811444187e-01, -7.101662020140024e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tfvw_opt_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.154388740626661e-02, 0.000000000000000e+00, 1.151273949201530e-02, 3.449246736686695e-02, 0.000000000000000e+00, 3.440454304308932e-02, 2.487890634348689e+00, 0.000000000000000e+00, 2.491067424669345e+00, 1.560414464473780e+01, 0.000000000000000e+00, 4.690057131491636e+04, 2.504729166250900e+02, 0.000000000000000e+00, 1.470171659578715e+09, 4.033351378532151e+04, 0.000000000000000e+00, 4.122651861079607e+04, 4.933449725883658e+09, 0.000000000000000e+00, 1.373122666672850e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
