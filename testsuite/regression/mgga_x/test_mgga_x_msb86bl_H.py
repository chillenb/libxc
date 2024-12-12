
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_msb86bl_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.722087309412942e-01, -6.124191377808744e-01, -3.696886311603943e-01, -1.334744773916906e-01, -1.635652629840941e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_msb86bl_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.957542043460383e-01, -8.833398646122426e-17, -7.907448763018889e-01, -1.555980780163193e-16, -4.494152642250065e-01, -2.734619578389743e-17, -1.156694919921859e-01, -8.749891659586644e-17, -1.523708977953147e-02, -2.301188979799802e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.039066051733715e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.199033080317941e-02, 0.000000000000000e+00, 0.000000000000000e+00, -9.500746767059463e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.012088651217539e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.251251783982146e+03, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.151846855677654e-12, 0.000000000000000e+00, 2.444569619433950e-11, 0.000000000000000e+00, 8.006261155831794e-11, 0.000000000000000e+00, -4.326414693093551e-16, 0.000000000000000e+00, 2.141063824778525e-08, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
