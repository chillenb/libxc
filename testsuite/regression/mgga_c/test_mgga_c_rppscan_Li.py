
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rppscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.565505986010903e-02, -2.438854299236690e-02, -1.481762904283233e-02, -2.012621562903424e-04, -3.846315914119575e-08, -1.067638503069928e-03, -5.788917053522302e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rppscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.481108043040237e-02, -3.473828000250047e-02, -3.781899381156490e-02, -3.775361074584977e-02, -4.931938850677482e-02, -4.933936153499957e-02, -1.594402626466464e-03, -1.591277537839823e-01, -1.343048660198849e-02, -9.393543880072111e-02, -2.008547655874807e-03, -2.023105915393410e-03, -1.075222065182060e-05, -1.347139219250738e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.683420483836390e-05, 3.366840967672778e-05, 1.683420483836390e-05, 7.432328808584583e-05, 1.486465761716917e-04, 7.432328808584583e-05, 1.843077645214984e-02, 3.686155290429969e-02, 1.843077645214984e-02, 2.112852474674841e+00, 4.225704949349682e+00, 2.112852474674841e+00, 7.837063043306027e+01, 1.567412608661205e+02, 7.837063043306027e+01, 2.991459126025925e+00, 5.982918252051848e+00, 2.991459126025925e+00, 6.472757710678005e+03, 1.294551542135601e+04, 6.472757710678005e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.563259570392184e-03, -1.563259570392183e-03, -2.251993729828261e-03, -2.251993729828260e-03, -7.795966562943723e-03, -7.795966562943719e-03, -8.030276864435672e-02, -8.030276864433900e-02, -1.874213984263762e-01, -1.874213982749572e-01, -1.738969408900175e-10, -1.738969408900175e-10, -1.666241956151331e-19, -1.666241956151332e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
