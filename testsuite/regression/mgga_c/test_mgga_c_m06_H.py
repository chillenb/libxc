
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.979350467510434e-12, 1.942830422052704e-02, -1.935244518188705e-02, -1.501398002024626e-01, -9.992483187937133e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [4.419056216454873e-02, -6.828912475654785e-01, 5.820599678141309e-02, -5.924381074725632e-01, 5.919756817444329e-02, -3.650319659083329e-01, -1.076969748168484e-01, 4.619694055013043e-01, -1.603207005299804e-03, 2.671423696347350e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.807535070161319e+00, 0.000000000000000e+00, 7.017008656830159e+17, 5.733012974103597e-02, 0.000000000000000e+00, 5.078587280916719e+17, 1.214986617705826e+00, 0.000000000000000e+00, 1.257023950853427e+17, 6.531342702692498e+02, 0.000000000000000e+00, -1.558103112997204e+18, 8.439266741976565e+07, 0.000000000000000e+00, -4.073347483648832e+17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.310943313940130e+00, 4.443985826753845e+06, -4.643733141323571e-02, 4.117610495573543e+06, -9.769288127967000e-02, 2.977197220233121e+06, -8.271132311512701e-02, 1.472105715027063e+05, 2.887927208119629e-04, -2.615291277697518e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
