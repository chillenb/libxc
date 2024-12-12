
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_bloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.042753887632236e-01, -6.321216493810448e-01, -3.706287527757041e-01, -1.314510561259297e-01, -7.396614691900474e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_bloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.300376375902854e-01, -6.324631335342487e-17, -7.983852442031789e-01, -2.444696923882759e-16, -4.858356846826324e-01, 5.935650800746030e-17, -1.257943941199588e-01, -4.453361195651999e-17, -9.855219278933926e-03, -7.627559464303277e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.455540242601702e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.505085357831538e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.411873193881620e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.521955181111875e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.540656997359904e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.062245550667457e+01, 0.000000000000000e+00, 9.464959818262575e-02, 0.000000000000000e+00, -2.249350853780075e-03, 0.000000000000000e+00, -4.906420848914150e-04, 0.000000000000000e+00, 4.528988642205197e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
