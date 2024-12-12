
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_lp90_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.935581000277548e-01, -4.932970509147474e-01, -2.860253867541708e-01, -8.098807956170472e-02, -6.125340658195137e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_lp90_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.961763391024407e-01, -6.961763391024406e-01, -6.351872709925913e-01, -6.351872709925913e-01, -3.700179337072254e-01, -3.700179337072254e-01, -8.221929376145413e-02, -8.221929376145413e-02, 4.826371828669831e-01, 4.826371828669831e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.885593549399931e-03, -3.771187098799862e-03, -1.885593549399931e-03, -2.918000925269055e-03, -5.836001850538111e-03, -2.918000925269055e-03, -2.495800575684099e-02, -4.991601151368198e-02, -2.495800575684099e-02, -4.911853475578242e+00, -9.823706951156485e+00, -4.911853475578242e+00, -9.975431925369909e+05, -1.995086385073982e+06, -9.975431925369909e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([5.621389481441295e-04, 5.555487043507060e-04, 6.271109775717650e-04, 6.168883621643038e-04, 1.073259770046247e-03, 1.069360144983213e-03, 4.023038582987931e-03, 4.022934364677557e-03, 8.542678590925457e-02, 8.542678657528807e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
