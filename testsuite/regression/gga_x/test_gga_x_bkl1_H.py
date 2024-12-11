
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bkl1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.226286043006586e-01, -6.058944809140545e-01, -4.042517560692773e-01, -1.538723326264232e-01, -4.101571076394583e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bkl1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.274920485942753e-01, -6.618071898855587e-17, -6.859662965994736e-01, -1.917176857448182e-16, -3.663235772683844e-01, -1.391027420457566e-17, -2.037354371079692e-01, -7.000904758030024e-17, -5.469104096308115e-03, -3.539622183030311e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bkl1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.109887045267865e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.661807526108039e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.771189520596682e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.606988133142245e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.737620087406433e-01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
