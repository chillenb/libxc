
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_regtpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.208863927633740e-02, -1.875338432262434e-02, -8.898355178054443e-03, -2.599113766272223e-04, -6.948119325982799e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_regtpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.693879935431634e-02, 1.721624345323563e+01, -4.117163228020032e-02, 7.103853482010346e+02, -2.876189302040136e-02, 4.161403937222119e+02, -1.490419151712521e-03, 5.429911640212540e+00, -4.506701542115674e-09, 4.283130421737971e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_regtpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.544568518854764e-02, 3.089137037709527e-02, 1.544568518854764e-02, 9.990141711516970e-03, 1.998028342303394e-02, 9.990141711516970e-03, 4.319387304877545e-02, 8.638774609755090e-02, 4.319387304877545e-02, 1.407109502670590e-01, 2.814219005341180e-01, 1.407109502670590e-01, 2.960104239720123e-03, 5.920208480833401e-03, 2.960104239720123e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
