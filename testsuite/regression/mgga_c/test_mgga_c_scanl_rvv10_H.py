
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scanl_rvv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.356646216685396e-02, -1.284587227669590e-02, -3.470933507616573e-04, -1.338096300429470e-13, -2.731084015689018e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scanl_rvv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.179376380137613e-02, 1.707621176055696e+01, -4.903082843714947e-02, 9.917932496924891e+01, -1.136505166126755e-02, 6.694140068450594e-01, 6.505838884834123e-14, -1.096289566994679e-01, -5.220037686364359e-04, -2.829707959547908e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_rvv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.725865911706671e-02, 3.476534203966912e-02, 1.738267101983456e-02, 1.639447928564843e-02, 3.064079914773073e-02, 1.532039957386537e-02, 2.400964575370531e-02, 1.010337130918268e-01, 5.051685654591340e-02, -3.865691661323135e-12, 2.335894399867393e+01, 1.167947199933696e+01, 1.444402696440874e+02, 6.330373358039341e+04, 3.165186679019671e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_rvv10_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-6.188162007478658e-04, 0.000000000000000e+00, -4.063962936345176e-03, 0.000000000000000e+00, -1.507519017075082e-03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.920548802178135e-05, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
