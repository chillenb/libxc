
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_r_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547400078540e-01, -5.689552237840139e-01, -3.460739192131563e-01, -1.288804253370593e-01, -1.521480835798320e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_r_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285213490659593e-01, -7.944846736781688e-17, -7.284454583105145e-01, -1.930906794484011e-16, -4.115585853284690e-01, -1.618343961711233e-17, -1.111985721877524e-01, -6.498484210620315e-17, -1.432994816388021e-02, -6.667587090585796e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_r_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.483982098314294e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.400971589991257e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.089199899401997e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.825860128124493e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.759680085770272e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
