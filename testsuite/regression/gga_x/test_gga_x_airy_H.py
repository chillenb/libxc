
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_airy_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.215766116602851e-01, -5.543795861143498e-01, -3.305524432496381e-01, -1.203007218929786e-01, -3.929670400707882e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_airy_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.289374585025269e-01, -1.084702196749642e-16, -7.385024759999679e-01, -1.889762048844721e-16, -4.034994681409911e-01, 2.943215011961694e-17, -9.692593085020201e-02, -5.044780559585815e-17, -9.855366976960730e-03, -4.770505059840833e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_airy_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.586763229777409e-03, 0.000000000000000e+00, 0.000000000000000e+00, -3.113501425934674e-04, 0.000000000000000e+00, 0.000000000000000e+00, -8.132340313330561e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.144746612398695e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.399297887518520e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
