
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_revssb_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.726122758298315e-01, -6.021899963356202e-01, -3.647936372032339e-01, -1.348152330149233e-01, -7.396595668463616e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_revssb_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.970631794323934e-01, -1.646453501753374e-17, -7.912013646510386e-01, -3.575768814858049e-16, -4.042722340217252e-01, 1.215816098264955e-17, -1.403454877687302e-01, -1.112724597323174e-16, -9.855122485521731e-03, -1.616234693715608e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_revssb_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.844465877499551e-03, 0.000000000000000e+00, 0.000000000000000e+00, -7.053390340661760e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.857056380257403e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.444191487150705e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.606117052033845e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
