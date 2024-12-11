
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lsrpbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220352060441524e-01, -5.779128128451245e-01, -3.613524194166389e-01, -1.413713105829657e-01, -1.396512581697563e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lsrpbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282809736730165e-01, -9.285074654880844e-17, -7.172093453109115e-01, -2.428773087003415e-16, -3.950532744275140e-01, -4.774538903223696e-17, -1.638012279050880e-01, -6.770506752457851e-17, -1.193675927392325e-14, -2.066778627143259e-32])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lsrpbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.686184598149283e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.477639026815296e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.894561322926987e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.779538930576519e+00, 0.000000000000000e+00, 0.000000000000000e+00, 9.389614766406015e-09, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
