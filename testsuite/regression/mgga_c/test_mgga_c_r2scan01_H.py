
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan01_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.355227538141662e-16, -7.441802950863864e-03, -2.301361388833259e-02, -2.144833086969269e-02, -2.668430861371849e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan01_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.035667529907872e-04, -2.199798757255405e-01, 7.132292917049254e-03, -2.248651304197484e-01, 6.924294726950508e-03, -1.953126949442377e-01, -2.350919027535014e-02, -6.720020452820050e-02, -3.402201066015068e-03, -2.631119543777522e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.236210832420324e-03, 8.472421664840654e-03, 4.236210832420324e-03, 1.703165489317377e-02, 3.406330978634756e-02, 1.703165489317377e-02, 2.826662254525892e-01, 5.653324509051785e-01, 2.826662254525892e-01, 3.957160309289886e+01, 7.914320618579772e+01, 3.957160309289886e+01, -3.023251810477306e+06, -6.046503620954611e+06, -3.023251810477306e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.010329761559207e-02, -9.984851465180170e-03, -1.981357844554831e-02, -1.949059479586212e-02, -3.987384361939933e-02, -3.972896439791220e-02, -2.228939262695857e-03, -2.228881521195472e-03, -3.110832447309604e-07, -3.110832471563338e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
