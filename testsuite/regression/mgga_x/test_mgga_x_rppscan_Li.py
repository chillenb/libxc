
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rppscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037410948453068e+00, -1.413071631087617e+00, -3.261026581357812e-01, -1.840458117349171e-01, -7.184056188003680e-02, -5.937164977314107e-03, -1.442334962107023e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rppscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.672621291720869e+00, -2.675104215431271e+00, -1.831014793470908e+00, -1.832684218797979e+00, -2.623452561785808e-01, -3.110609988193890e-01, -2.432149035741976e-01, 3.100464165747890e-01, -8.240571809394774e-02, 3.397592853568917e-03, -9.521296786738739e-03, 3.235860568481085e-01, -4.692758857734685e-05, -3.660998326213039e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rppscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.073133145434802e-04, 0.000000000000000e+00, -2.065636204707450e-04, -8.116239690247084e-04, 0.000000000000000e+00, -8.080639543183398e-04, -2.510209331105407e-01, 0.000000000000000e+00, -1.844452303254443e-01, -3.311832407365086e+00, 0.000000000000000e+00, -8.200972651164244e+03, -9.010493207395648e+01, 0.000000000000000e+00, -7.088046755258915e+06, 2.043255686462311e+01, 0.000000000000000e+00, -7.331944745012736e+03, 3.271621446842214e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rppscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.113348702229148e-02, 1.112325443458508e-02, 1.488745896516672e-02, 1.486288902547143e-02, 6.327271267487847e-02, 4.730319672620203e-02, 1.293861119264151e-01, 1.050986093253013e-01, 2.327293278665363e-01, 2.898768561085147e-03, 1.087105851013621e-10, 1.068954787883535e-01, 1.230165639471858e-19, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
