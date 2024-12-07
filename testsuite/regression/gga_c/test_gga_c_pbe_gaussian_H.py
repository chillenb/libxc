
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_gaussian_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.206368534054605e-02, -1.817854332355233e-02, -8.087198355320964e-03, -1.585941610084737e-04, -2.329695032807916e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_gaussian_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.697774809624417e-02, 1.597032959843976e+00, -4.114821518572898e-02, 7.387593192825983e+01, -2.759462978747896e-02, 4.139495338810676e+01, -9.424600992699109e-04, 3.382817802983852e-01, -1.515737550880289e-09, 1.066938797814748e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_gaussian_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.644070770466411e-02, 3.288141540932821e-02, 1.644070770466411e-02, 1.020699084687689e-02, 2.041398169375378e-02, 1.020699084687689e-02, 4.208666023717062e-02, 8.417332047434126e-02, 4.208666023717062e-02, 8.810650490377328e-02, 1.762130098075465e-01, 8.810650490377328e-02, 9.926733051752769e-04, 1.985346609653593e-03, 9.926733051752769e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
