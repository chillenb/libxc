
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbr_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.946810104544020e-01, -5.650005866957087e-01, -3.635050073267185e-01, -1.456299300425445e-01, -4.997488031487943e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbr_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.910101606175381e-01, -2.270936948492601e-18, -6.765927942591679e-01, -1.689410611271705e-16, -3.822394757321777e-01, -2.333565531157715e-17, -1.240550758242117e-01, -7.517906472998792e-17, -2.086641411672994e-02, -1.337382969409487e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.769096012287726e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.252209715519341e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.973900423152260e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.541053179221676e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.924419039250922e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-5.417301023743228e-03, 0.000000000000000e+00, -8.592246508685735e-03, 0.000000000000000e+00, -1.448670824779073e-02, 0.000000000000000e+00, -1.416827113691408e-02, 0.000000000000000e+00, -8.031568764602584e-03, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
