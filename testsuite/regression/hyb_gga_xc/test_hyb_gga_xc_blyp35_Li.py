
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_blyp35_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_blyp35", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.226982264168475e+00, -8.885396538877051e-01, -2.667981325204263e-01, -1.043961264410604e-01, -5.282059209178111e-02, -8.866187537964772e-02, -3.487913268926385e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_blyp35_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_blyp35", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.521425715940204e+00, -1.522635824849973e+00, -1.055751863115060e+00, -1.056467602459320e+00, -3.251893648959858e-01, -3.255323145701082e-01, -1.333181767885742e-01, -1.055122954841231e-01, -4.767537125273098e-02, -3.532022938685628e-02, -2.651990710110483e-02, -2.672542893155516e-02, -4.870600812254084e-03, -4.288122186878192e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_blyp35_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_blyp35", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.714781265594647e-04, 5.222815421851711e-06, -1.710131868854779e-04, -6.292999745517865e-04, 3.646941789248587e-05, -6.278568769147665e-04, -3.993303976900453e-02, 4.773762863586187e-02, -3.975270969047276e-02, -2.866531625219813e+00, 4.596134769453040e+00, -8.672848216083121e+02, -4.978686496586535e+01, 2.356939734329661e+01, -3.152695378771678e+07, -7.571226215349819e+02, 7.936097321777658e-02, -7.583306826420845e+02, -9.360014537709148e+07, 0.000000000000000e+00, -2.788247295488250e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
