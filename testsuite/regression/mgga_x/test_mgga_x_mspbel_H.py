
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mspbel_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.788635340315492e-01, -6.176273354350585e-01, -3.716899522112209e-01, -1.266659045116731e-01, -7.394596220822191e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mspbel_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.046661866564328e-01, -7.852404673825092e-17, -7.998178842852960e-01, -3.166866022930637e-16, -4.564149803421946e-01, 4.173638005741870e-18, -1.275659210594243e-01, -7.638953810244065e-17, -9.847161961891194e-03, -3.658278748240440e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.442213976752759e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.100155046403519e-02, 0.000000000000000e+00, 0.000000000000000e+00, -8.554822225756370e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.651196952999865e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.828530242751121e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.304075912104429e-12, 0.000000000000000e+00, 2.741111126266403e-11, 0.000000000000000e+00, 8.797582451366135e-11, 0.000000000000000e+00, -3.451612664743065e-16, 0.000000000000000e+00, 4.714276835924704e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
