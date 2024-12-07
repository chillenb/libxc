
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ft97_b_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219804506682398e-01, -5.755814997168162e-01, -3.585172853364379e-01, -1.647538784341599e-01, -6.301633985380936e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ft97_b_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.283534759782025e-01, -1.698424103589198e-16, -7.191886827364934e-01, -2.522313024227552e-16, -3.939909795020435e-01, -2.601167567759424e-18, -9.602506765375587e-02, -8.197650008628678e-17, -1.730814823383200e-02, -1.532410446319998e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ft97_b_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.462992023920292e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.241317958125658e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.835204545525792e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.391767470834503e+01, 0.000000000000000e+00, 0.000000000000000e+00, -5.330941675837137e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
