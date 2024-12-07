
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_revtca_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.463715987925622e-02, -1.945115517076845e-05, -7.001471814363270e-09, -2.142620999014538e-08, -2.235615303265301e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_revtca_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.127021926238338e-02, -1.528083426159034e+03, -1.291626704242456e-03, -1.236738353366429e+00, -1.596505849525020e-05, -7.274402201598183e-02, 5.000788591375675e-06, -2.642999492163970e-03, 1.340707245577820e-09, -1.807574372718828e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_revtca_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.246452718617699e-02, 1.449290543723542e-01, 7.246452718617699e-02, 7.840477191731945e-04, 1.568095438346389e-03, 7.840477191731945e-04, 4.645637782355546e-05, 9.291275564711092e-05, 4.645637782355546e-05, -7.569719310823013e-04, -1.513943862164602e-03, -7.569719310823013e-04, -1.927743214996946e-03, -3.855486429993891e-03, -1.927743214996946e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
