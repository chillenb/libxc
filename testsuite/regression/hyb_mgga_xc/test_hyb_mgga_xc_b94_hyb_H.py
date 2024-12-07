
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b94_hyb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.241897287838675e-01, -4.170324410183441e-01, -3.103721998495291e-01, -1.405379138663160e-01, -6.361504451383984e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b94_hyb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.827188199105641e-01, -2.392922991847929e-02, -6.248494314396850e-01, -1.806485216012224e-02, -3.816198073839967e-01, -7.980938952345347e-03, -1.113090075132789e-01, -1.161586102515430e-03, -2.815835607795000e-02, -8.130554931076549e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.522758355484872e-03, 0.000000000000000e+00, 0.000000000000000e+00, 9.871281337660012e-04, 0.000000000000000e+00, 0.000000000000000e+00, -1.679199090016801e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.477321622147633e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.075846572090512e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.008856760265980e-04, 0.000000000000000e+00, -2.689171929537179e-03, 0.000000000000000e+00, -1.688225958592482e-02, 0.000000000000000e+00, -1.225940910401072e-02, 0.000000000000000e+00, -5.273526354832206e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.555667825493980e-02, 0.000000000000000e+00, -1.697158855828679e-03, 0.000000000000000e+00, 5.776797543110648e-02, 0.000000000000000e+00, 4.244184388618781e-02, 0.000000000000000e+00, 2.107254587079202e-02, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
