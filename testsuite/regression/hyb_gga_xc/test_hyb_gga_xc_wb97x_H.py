
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.232898938907165e-01, -3.610234880731627e-01, -1.758286757900803e-01, -3.495451519598924e-02, -7.043087401272913e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.055017206062171e-01, -2.601063679264922e-01, -5.193157799552180e-01, -2.642046594412210e-01, -2.202434941057854e-01, -2.191690823572539e-01, 6.879774260797319e-03, 5.552069458816358e-02, -9.084854014047974e-04, 3.675531018606934e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.333880807337996e-01, 0.000000000000000e+00, -3.854719753735396e+20, 2.802993658715983e-04, 0.000000000000000e+00, -2.468559987856915e+20, -1.043688053131982e-01, 0.000000000000000e+00, -4.579692225279492e+19, -7.771351508063373e+00, 0.000000000000000e+00, 7.333328549981778e+19, -4.679159381832296e-01, 0.000000000000000e+00, 1.003083468872517e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
