
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_ccalda_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.247092247992547e-02, -3.118072389969410e-02, -2.518506028185151e-02, -1.327195891566465e-02, -6.413831866940176e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_ccalda_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.561495538385070e+00, 2.335234816230650e+00, 1.421083002488279e+02, 1.418928086705681e+02, 3.566899781039742e+02, 3.565235968452450e+02, 2.685034909113888e+03, 2.684960108112937e+03, 3.219154902042374e+04, 3.219154323545337e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.062641491968992e+02, -2.125282983937985e+02, -1.062641491968992e+02, -1.760644569080359e+02, -3.521289138160718e+02, -1.760644569080359e+02, -2.077466531004066e+03, -4.154933062008132e+03, -2.077466531004066e+03, -8.059444626485293e+05, -1.611888925297058e+06, -8.059444626485293e+05, -6.859617994207877e+10, -1.371923598841575e+11, -6.859617994207877e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.534383598161369e+02, 2.533106457243127e+02, 3.027057400321001e+02, 3.006773565219137e+02, 7.146921186146603e+02, 7.132559139636990e+02, 5.280850799155905e+03, 5.280819166997137e+03, 4.699506725679847e+04, 4.699506752644513e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
