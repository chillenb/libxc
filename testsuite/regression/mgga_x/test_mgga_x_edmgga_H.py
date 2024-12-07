
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_edmgga_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.978120061884470e-01, -4.950606127313450e-01, -3.649821711729853e-01, -1.710455323441834e-01, -6.975200665449983e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_edmgga_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.795816949322226e-01, -1.069783124328887e-16, -7.628021465769189e-01, -6.807685519001890e-17, -4.511944063829621e-01, -9.322519076563175e-17, -1.296021201805525e-01, -3.239618218134848e-17, -3.185398896859657e-02, -2.009231150084959e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.578684869766867e-04, 0.000000000000000e+00, 0.000000000000000e+00, -1.020794415685275e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.876476124897503e-01, 0.000000000000000e+00, 0.000000000000000e+00, -9.277008106527081e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.321892610290989e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-9.412847773309401e-05, 0.000000000000000e+00, -4.387602336768712e-03, 0.000000000000000e+00, -1.613867994042509e-02, 0.000000000000000e+00, -1.519661029498078e-02, 0.000000000000000e+00, -5.689550239873116e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.765139109323761e-04, 0.000000000000000e+00, 1.755040934707485e-02, 0.000000000000000e+00, 6.455471976170038e-02, 0.000000000000000e+00, 6.078644117992311e-02, 0.000000000000000e+00, 2.275820095949246e-02, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
