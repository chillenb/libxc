
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpssh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.669235369259667e+00, -1.172303946522198e+00, -3.175698140076465e-01, -1.608082993092685e-01, -6.806820523327956e-02, -1.849570635908488e-02, -3.155077402190087e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpssh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.217257836490357e+00, -2.219015701050922e+00, -1.557919236089814e+00, -1.559058783848908e+00, -4.108995074687208e-01, -4.106280961609957e-01, -2.122168911863504e-01, -1.478150092543545e-01, -8.757702690989536e-02, -7.227410006044778e-02, -2.475635134162671e-02, -2.453390659956219e-02, -4.987407775606612e-04, -2.034027331084167e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpssh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.705034565139385e-04, 2.772505973492434e-04, -1.681052578198748e-04, -2.686729888937808e-04, 1.193829360748139e-03, -2.668662361648374e-04, 1.438742917672143e-01, 3.593578553367118e-01, 1.432343397306134e-01, -2.542384902885670e+00, 8.341737401120239e+00, 3.920623792161559e+00, 1.388604631166015e+02, 3.363362362756724e+02, 1.665692697888426e+02, -1.064984872882330e-04, 1.079340474789106e-08, -2.374912549005423e-01, -7.297554088826198e-11, -1.923003370021578e-15, -1.401658077457393e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpssh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.091309085606080e-03, 2.074334129925319e-03, 2.127730659117456e-03, 2.131514788769633e-03, -5.719984526675045e-04, -6.053478767255040e-04, 2.414967217294566e-02, -3.192826675701848e-10, -1.400949304986850e-02, 3.129368614104974e-17, 6.817654276745144e-16, 6.927342075933655e-11, 1.283013649711022e-33, -4.945573789409534e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
