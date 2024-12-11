
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_revtpssh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.669235132977994e+00, -1.172303946522198e+00, -3.175698140076465e-01, -1.608083068349435e-01, -6.806820523328719e-02, -1.849260291612813e-02, -3.155077402183444e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_revtpssh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.217257456862524e+00, -2.219015010053682e+00, -1.557919236089814e+00, -1.559058783848908e+00, -4.108995074687208e-01, -4.106280961609957e-01, -2.122168767792301e-01, -1.477929582237409e-01, -8.757702690988413e-02, -7.227409827040937e-02, -2.475581865554852e-02, -2.450939792054933e-02, -4.987407775570555e-04, -2.034027331084220e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.691352139837293e-04, 2.718091272326938e-04, -1.664105854325999e-04, -2.773542132443534e-04, 1.160760060229794e-03, -2.755288863136644e-04, 1.457397078656040e-01, 3.248165986340941e-01, 1.450181345528594e-01, -3.180336359925781e+00, 6.898877508101942e+00, 3.057689139569888e+00, 9.925996071401156e+01, 2.447101468657844e+02, 1.198482034592178e+02, -1.669707593371956e-04, 2.843160860021923e-08, -3.717478169766382e-01, -1.144192079461949e-10, 2.024865339056517e-15, 8.287671039578362e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.091124146121111e-03, 2.074334297919063e-03, 2.127730659117456e-03, 2.131514788769633e-03, -5.719984526675045e-04, -6.053478767255040e-04, 2.414967217303154e-02, -2.733267949905852e-10, -1.400949304986850e-02, 5.454994136652699e-17, 1.188646831433853e-15, 1.219195049825388e-10, 2.236447806886262e-33, -4.945573789405781e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
