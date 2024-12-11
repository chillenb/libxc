
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_c09x_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.217952513626031e-01, -5.661330604882441e-01, -3.417129175215689e-01, -1.337285017191777e-01, -9.208004306135248e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_c09x_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.286004583847326e-01, -6.353265334974167e-17, -7.316375490341476e-01, -2.243615059046933e-16, -4.145474742011327e-01, 1.115386045302591e-17, -8.618232100122564e-02, -7.003785090552306e-17, -1.227733921821159e-02, -1.893998708328086e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_c09x_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.053910456079778e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.077920005232447e-02, 0.000000000000000e+00, 0.000000000000000e+00, -8.969360442507418e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.036928720066080e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.551557459717164e-43, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
