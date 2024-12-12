
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pw6b95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.354546513897435e+00, -9.812972907401395e-01, -3.258984174837740e-01, -1.163442394922345e-01, -5.992932642560635e-02, -5.789750503790081e-04, -1.221316061300120e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pw6b95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.668768752484872e+00, -1.670092644228721e+00, -1.152390508539972e+00, -1.153174034597005e+00, -3.030595123640002e-01, -3.030206231049525e-01, -1.490130090616449e-01, -2.898531886005789e-03, -5.603576322750260e-02, -1.616875528940094e-06, -2.398550212163443e-03, -2.172179762779519e-03, -8.246193329593103e-06, -2.389944179704456e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw6b95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.967376180913709e-04, 0.000000000000000e+00, -1.961502252975314e-04, -7.072251865519368e-04, 0.000000000000000e+00, -7.053524492052289e-04, -6.149553203052285e-02, 0.000000000000000e+00, -6.148474415650253e-02, -1.357197470961532e+00, 0.000000000000000e+00, 1.904198327145111e+01, -4.791209900270686e+01, 0.000000000000000e+00, 8.837753537558932e+02, 1.357844842249961e+01, 0.000000000000000e+00, 1.181397439029621e+01, 1.916864449346175e+04, 0.000000000000000e+00, 1.135369149546540e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw6b95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.564560006295679e-03, -1.563656315570130e-03, -1.807818732307285e-03, -1.807005275984981e-03, -4.707594838529174e-04, -4.678102123367323e-04, -9.554099317920803e-02, -1.262614296020221e-07, -1.653393519549892e-02, -4.137508929756903e-11, -1.473706059727646e-07, -1.348695534096974e-07, -9.178070520725264e-12, -4.795501805496794e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
