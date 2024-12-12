
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.131066891153007e-01, -6.411006708828517e-01, -3.738406206760299e-01, -1.245579374958008e-01, -7.395139413207376e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.494203103414690e-01, -1.805605187528712e-16, -8.179730556486019e-01, -2.313674640240462e-16, -4.668541873761110e-01, -2.706259086564295e-17, -1.144734721599907e-01, -5.706638462196558e-17, -9.849325228311615e-03, -8.600684131045478e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.959238832647987e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.648155371771970e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.917360230930993e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.952434314211827e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.678678621589832e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.079651746427658e-01, 0.000000000000000e+00, 8.080768099889027e-02, 0.000000000000000e+00, 6.755129232836744e-02, 0.000000000000000e+00, 1.508781248401447e-03, 0.000000000000000e+00, 7.987585103494488e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
