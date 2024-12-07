
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_b00_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.015897654815789e-01, -6.147156936900496e-01, -3.861432811170180e-01, -1.565268722366890e-01, -7.519594662151098e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_b00_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.882491662357947e-01, -1.267767358984705e-19, -8.146574384862004e-01, -2.884899061531067e-16, -7.460212691616642e-01, -1.296908713897936e-16, -9.543739663186049e-02, -6.686972566435900e-17, -3.326658189753259e-02, -3.647461902707189e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.000748214685617e-04, 0.000000000000000e+00, 0.000000000000000e+00, -9.222189807011458e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.442144044086041e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.335394356176817e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.639471324395632e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.192938422241143e-04, 0.000000000000000e+00, -3.963903105818277e-03, 0.000000000000000e+00, -2.100372105617534e-02, 0.000000000000000e+00, -1.365415856397491e-02, 0.000000000000000e+00, -6.233481143423426e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.477905934107365e-02, 0.000000000000000e+00, -1.090637970420154e-01, 0.000000000000000e+00, 4.173577937178067e-01, 0.000000000000000e+00, 2.535120700964671e-02, 0.000000000000000e+00, 2.493387015952915e-02, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
