
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.368382464689878e-02, -7.867738050348934e-02, -3.585592086280753e-01, -1.230124342137529e-02, -6.732924926771446e-02, -3.003774263458376e-02, -7.453872414602619e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.068221013955378e-02, -7.057205411548437e-02, -2.700101035705501e-02, -2.687064918366853e-02, -1.031885932336603e-01, -1.035032898117028e-01, -1.362157167616426e-02, -8.387822411800322e-02, -7.182852499299596e-03, -3.661129109589216e-01, -3.775011575320138e-02, -3.817379232380904e-02, -8.768713813899684e-04, -1.286685901878291e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.574990834760380e-04, -1.514998166952076e-03, -7.574990834760380e-04, -9.710379919247884e-04, -1.942075983849577e-03, -9.710379919247884e-04, 1.800145731273688e+00, 3.600291462547377e+00, 1.800145731273688e+00, -2.234404928276840e+01, -4.468809856553681e+01, -2.234404928276840e+01, 1.619118196018414e+03, 3.238236392036828e+03, 1.619118196018414e+03, 1.090673410405362e-07, 2.181346879858812e-07, 1.090673410405362e-07, 1.148799770798159e-15, -2.032112300103491e-14, 1.148799770798159e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.482497985274037e-04, 1.482497985273911e-04, -8.771375468467790e-03, -8.771375468467789e-03, -5.634841276891169e-02, -5.634841276891156e-02, -1.823948184518310e-02, -1.823948184518010e-02, -6.139749904804859e-01, -6.139749900572670e-01, -9.004559906192447e-08, -9.004559906319243e-08, -2.379411459816089e-19, -2.379573016813632e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
