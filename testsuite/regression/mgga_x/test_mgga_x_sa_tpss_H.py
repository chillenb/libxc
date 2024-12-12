
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_sa_tpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.042740067055809e-01, -6.321204037401396e-01, -3.706280292207176e-01, -1.314482220397800e-01, -7.396280283613807e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_sa_tpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.293290098131638e-01, -5.855414478692833e-17, -7.730052822683187e-01, -3.009377296163672e-16, -4.598233891067255e-01, 3.998197500797494e-17, 7.409862448348832e-02, -6.447859894731261e-17, 1.421163029353586e+01, -2.365423333277441e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.484450098033815e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.648537624771763e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.656047954028810e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.552188824561860e+01, 0.000000000000000e+00, 0.000000000000000e+00, -3.030421653016157e+07, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.069140519706822e+01, 0.000000000000000e+00, 1.486945847097128e-01, 0.000000000000000e+00, 4.986517682366224e-02, 0.000000000000000e+00, 3.926561848580149e-01, 0.000000000000000e+00, 2.076133712175270e+01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
