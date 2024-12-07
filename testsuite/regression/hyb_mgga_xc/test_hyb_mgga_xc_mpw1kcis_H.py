
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpw1kcis_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.286399183427462e-01, -4.933719286212540e-01, -3.073824503445168e-01, -1.213369251839308e-01, -3.256098407104042e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpw1kcis_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.359410479762732e-01, -2.236903732310476e-01, -6.321241249309393e-01, -1.870772614507488e-01, -3.570760532260106e-01, -1.295847842555173e-01, -1.022663452721873e-01, -2.851872973089984e-02, -1.204367396310769e-03, -9.920875263050760e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1kcis_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.293023748718058e+00, 4.142665982842431e-02, 8.608116618147828e+13, 4.442724880646298e-03, 1.331120876263109e-02, 8.596311204654973e+13, -6.901063664846789e-02, 6.124766703732045e-02, 8.612673461976172e+13, -6.329129857490317e+00, 8.818506781180349e-01, 8.613032435712288e+13, 6.217741855922644e+02, 2.009733486597134e+01, 8.613054221073778e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1kcis_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1kcis_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.114511462711588e+00, -3.995554316157371e-04, -4.506598657184718e-02, -3.990074693494971e-04, -2.544624488301368e-02, -3.997669429258967e-04, -3.879145572284060e-03, -3.997836051175632e-04, -6.913882799401646e-06, -3.997846163109039e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
