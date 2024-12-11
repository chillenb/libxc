
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_jrgx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.224058370811983e-02, -2.245312517511208e-02, -1.240796872405720e-02, -4.462320378767620e-04, -7.363543947768586e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_jrgx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.675028405129285e-02, 7.777875047845768e-01, -4.153146815070007e-02, 4.785038155557844e+01, -3.308429020886354e-02, 3.708827835204593e+01, -2.515144355215311e-03, 8.766899102458282e-01, -4.790247424418803e-09, 3.371693189035095e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_jrgx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([9.352211864237405e-03, 1.870442372847481e-02, 9.352211864237405e-03, 8.077681963730753e-03, 1.615536392746151e-02, 8.077681963730753e-03, 4.364117326537326e-02, 8.728234653074651e-02, 4.364117326537326e-02, 2.321165742757311e-01, 4.642331485514625e-01, 2.321165742757311e-01, 3.137092467137959e-03, 6.274184934573525e-03, 3.137092467137959e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
