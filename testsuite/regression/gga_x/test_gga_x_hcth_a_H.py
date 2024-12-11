
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hcth_a_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.824151424638513e-01, -5.962003836204821e-01, -3.496448351465187e-01, -1.730242177328983e-01, -1.414649769628707e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hcth_a_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.114884169689813e-01, -7.775829933941734e-17, -8.141503057943221e-01, -2.093216493701156e-16, -4.374435485345093e-01, 4.680371955111937e-17, -2.904512603660843e-02, -9.863658283787037e-17, -2.670230287587340e-02, -1.407520518507138e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hcth_a_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.456584927157683e-02, 0.000000000000000e+00, 0.000000000000000e+00, 8.925852448829095e-03, 0.000000000000000e+00, 0.000000000000000e+00, -6.278714047839189e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.269814582988138e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.293848840738436e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
