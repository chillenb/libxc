
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_cap_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220233267220278e-01, -5.746928057924838e-01, -3.539519011287733e-01, -1.525739792431125e-01, -2.167293089003393e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_cap_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.283118046766146e-01, -1.013299318699113e-16, -7.240086953477693e-01, -2.159895439930718e-16, -4.068058814155579e-01, -4.739668504152211e-17, -7.084093426338824e-02, -7.946064281197444e-17, 4.850876589357560e-02, -2.485354583975815e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_cap_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.614598745269364e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.962394627087732e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.422395653165606e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.492444107608663e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.696738649523361e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
