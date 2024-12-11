
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_fr_pw86_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.034920357486888e+00, 1.691255905669205e+00, 6.217008294851558e-01, 6.144045622295567e-02, 4.307401361461745e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_fr_pw86_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.389073879902306e+00, 7.917415138003850e-16, 2.649273636250950e+00, 1.962486220695353e-17, 8.866514145182519e-01, 4.579085278528555e-17, 7.936884129949227e-02, 2.594925790047249e-17, 4.889293892185935e-04, -5.099407192476024e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_fr_pw86_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.773395043972348e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.872463479125162e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.265343995246772e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.592471739337756e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.829656184522710e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
