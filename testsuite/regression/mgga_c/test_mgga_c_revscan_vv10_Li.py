
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revscan_vv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.000044856604748e-02, -6.848797592369946e-02, -7.815300349796314e-02, -3.793740419020744e-03, -1.859981468143330e-02, 3.425369967853092e-05, 1.436024192532550e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revscan_vv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.312416881989856e-02, -1.294424542803383e-02, -3.536873336969776e-03, -3.293909536272310e-03, -7.621402762759989e-02, -7.628416762622761e-02, 2.312941479250950e-03, -1.529958424898711e-01, -1.022133578718869e-02, -5.047052871635668e-02, 2.604301512005362e-06, -1.137759244977571e-06, 2.943778298414299e-07, 2.470216952588227e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.023824283874695e-04, 2.047648567749390e-04, 1.023824283874695e-04, 7.284036640433407e-04, 1.456807328086681e-03, 7.284036640433407e-04, 3.259469181955202e-01, 6.518938363910406e-01, 3.259469181955202e-01, 5.436504203280996e+00, 1.087300840656199e+01, 5.436504203280996e+00, 2.497453361152924e+02, 4.994906722305848e+02, 2.497453361152924e+02, 1.094243721633776e-02, 2.188487443267553e-02, 1.094243721633776e-02, -1.121841465298950e-03, -2.243682930597899e-03, -1.121841465298950e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.747181861324744e-03, -5.747181861324740e-03, -1.047934063389493e-02, -1.047934063389492e-02, -2.689062903763840e-03, -2.689062903763838e-03, -1.812388091348041e-01, -1.812388091347640e-01, -1.006432046911086e-01, -1.006432046097982e-01, -4.367230058396611e-07, -4.367230058396624e-07, -2.347250249147314e-23, -2.347295308531787e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
