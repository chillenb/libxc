
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scan_rvv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.600178323180046e-13, -2.108459240535154e-12, -2.911920704562476e-12, -1.338096300429470e-13, -1.003113595139660e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scan_rvv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.035095944752623e-04, -2.199798184847155e-01, -3.945003182470149e-03, -2.101331087912539e-01, -8.151396840467836e-03, -1.813382331950551e-01, -3.891083697104631e-02, -1.096289566981773e-01, -8.495607211406681e-01, -8.512344646616125e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_rvv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.233872876609375e-03, 8.467745753218750e-03, 4.233872876609375e-03, 4.886437918149604e-03, 9.772875836299207e-03, 4.886437918149604e-03, 4.747229148401167e-02, 9.494458296802334e-02, 4.747229148401167e-02, 1.167947199974533e+01, 2.335894399949066e+01, 1.167947199974533e+01, 1.810307961099859e+06, 3.620615922199719e+06, 1.810307961099859e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_rvv10_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.009772162679788e-02, -9.979340846568327e-03, -8.401200514419983e-03, -8.264251481646829e-03, -1.633146530471571e-02, -1.627212590413835e-02, -7.652828687578785e-02, -7.652630438206089e-02, -1.240237345676739e+00, -1.240237355346301e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
