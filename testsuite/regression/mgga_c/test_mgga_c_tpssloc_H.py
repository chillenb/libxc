
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpssloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.920982555160916e-11, -1.890375728947350e-11, -2.939693084315867e-12, -1.294605225965901e-16, -1.202015424666020e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpssloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.447452252858847e-02, -6.447451309688269e-02, -3.538295896280663e-02, -3.538273398982787e-02, -8.235712402762703e-03, -8.235658777482266e-03, -1.060337023587885e-05, -1.060337005246243e-05, -2.055372197950336e-14, 1.436896495918554e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.637213804697255e+00, 5.274427609394513e+00, 2.637213804697255e+00, 4.382674084438772e-02, 8.765348168877540e-02, 4.382674084438772e-02, 4.796333030920289e-02, 9.592666061840606e-02, 4.796333030920289e-02, 3.182706294798187e-03, 6.365412589550391e-03, 1.252614639213007e-02, 4.372541300984566e-08, -1.503570845108076e-04, 6.466946908995474e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.289714312515355e+00, -6.285196535013992e+00, -7.535084730591353e-02, -7.516735300302166e-02, -1.650039295688134e-02, -1.649947889330692e-02, -2.085428693726420e-05, -2.085422077982690e-05, -2.994276749638758e-14, -2.994276749435383e-14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
