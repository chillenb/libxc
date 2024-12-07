
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw91_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219274908288185e-01, -5.801390273870550e-01, -3.611337645295494e-01, -1.365451424901111e-01, -6.501527958482537e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw91_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282595248580088e-01, -1.101258790458799e-16, -7.190838714819825e-01, -2.732441342577134e-16, -4.057969361543847e-01, -3.138471432191214e-17, -1.375561028541597e-01, -6.891197878897139e-17, -2.649799673180511e-04, -8.139921863471124e-21]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw91_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.498789476382988e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.528443071135728e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.653559849784720e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.009377977371516e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.424699964546556e+02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
