
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ev93_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.218572385263603e-01, -5.746618503426149e-01, -3.671525784500829e-01, -1.748963754138717e-01, -6.357563847648473e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ev93_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.285113823701145e-01, 1.409735367398576e-17, -7.091100922338558e-01, -3.277331836712492e-16, -3.584070990800675e-01, -3.209739393031888e-17, -1.711044099310785e-01, -7.009268104232142e-17, -8.561916777839598e-03, -9.679424721878472e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ev93_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.687956919783044e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.652502593787615e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.863784947750019e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.988932751876526e+00, 0.000000000000000e+00, 0.000000000000000e+00, 6.805340317742836e+01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
