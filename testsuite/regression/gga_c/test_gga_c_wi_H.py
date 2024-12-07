
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wi_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.714541361256700e-02, -4.300654062732721e-02, -1.329753876525956e-02, 2.762186894571208e-04, -8.373574792724647e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wi_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.300015210578792e-02, -7.300015210578792e-02, -8.446640881796752e-02, -8.446640881796752e-02, -5.914954124569498e-02, -5.914954124569498e-02, 1.060721707704760e-03, 1.060721707704760e-03, -5.024144168467094e-10, -5.024144168467094e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wi_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.896220506855445e-03, 9.792441013710891e-03, 4.896220506855445e-03, 1.363678331211938e-02, 2.727356662423876e-02, 1.363678331211938e-02, 9.108378084817614e-02, 1.821675616963523e-01, 9.108378084817614e-02, -7.794423117397117e-02, -1.558884623479423e-01, -7.794423117397117e-02, 3.126178287888970e-04, 6.252356575777940e-04, 3.126178287888970e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
