
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_oc2_n_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220349299338446e-01, -5.771807613072377e-01, -3.579294389863776e-01, -1.249453011476793e-01, -6.509433922202427e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_oc2_n_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282820768594884e-01, -7.988921729015008e-17, -7.199556659449223e-01, -2.403756264410595e-16, -4.063030208607768e-01, -1.666627922787795e-17, -1.404047987893297e-01, -5.500325950353268e-17, -8.675545528724153e-03, -4.600250372331970e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_oc2_n_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.683927764169799e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.304737793009102e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.549200173743830e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.947825426245044e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.956431416589575e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
