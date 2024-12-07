
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.407227376966134e+00, -1.017622999888826e+00, -3.100184734007570e-01, -1.204551057205506e-01, -6.094683666988105e-02, -1.019824425728517e-01, -4.024053191707456e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.745663578190857e+00, -1.747085680015334e+00, -1.208234876086760e+00, -1.209086649347536e+00, -3.591632455598308e-01, -3.594936336303509e-01, -1.538252411403466e-01, -1.090943582620196e-01, -5.501004332167748e-02, -3.609265478770473e-02, -3.018644308622384e-02, -3.040957173874610e-02, -5.616724496884501e-03, -4.933432001211269e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.988749797963847e-04, 5.222815421851711e-06, -1.983194227980574e-04, -7.323512482245722e-04, 3.646941789248587e-05, -7.305854767553424e-04, -5.140131724377221e-02, 4.773762863586187e-02, -5.121761248057796e-02, -3.307471956697132e+00, 4.596134769453040e+00, -1.001243801478670e+03, -5.744638298806392e+01, 2.356939734329661e+01, -3.637725708998859e+07, -8.736093124381027e+02, 7.936097321777658e-02, -8.750032613984209e+02, -1.080001677427979e+08, 0.000000000000000e+00, -3.217208417871057e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
