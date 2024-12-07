
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optpbe_vdw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219525350629000e-01, -5.737709452602829e-01, -3.541824042688255e-01, -1.390989240474909e-01, -8.394940109089572e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optpbe_vdw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.283911290281504e-01, 6.231586811609222e-18, -7.225002867895470e-01, -2.300048707089748e-16, -4.031270510614752e-01, 3.052799101410261e-17, -1.240956091134364e-01, -8.539671348369845e-17, -1.117934948453083e-02, -4.180859497835062e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optpbe_vdw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.348145281154954e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.975366027540133e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.509450912031403e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.907761904384117e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.111048356637478e+01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
