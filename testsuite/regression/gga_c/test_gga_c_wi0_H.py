
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wi0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.033740655740521e-02, -4.240583548180431e-02, -1.784941696489716e-02, 2.369667101642652e-04, -7.695897221939535e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wi0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.232054178416409e-02, -5.232054178416409e-02, -6.976045676799976e-02, -6.976045676799976e-02, -7.688271336304672e-02, -7.688271336304672e-02, 6.010988569370628e-04, 6.010988569370628e-04, -4.617538239506604e-10, -4.617538239506604e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wi0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.022754461291258e-03, 6.045508922582516e-03, 3.022754461291258e-03, 1.148691682034172e-02, 2.297383364068344e-02, 1.148691682034172e-02, 1.213208408225031e-01, 2.426416816450062e-01, 1.213208408225031e-01, -3.211474809149309e-02, -6.422949618298618e-02, -3.211474809149309e-02, 2.869824762166355e-04, 5.739649524332710e-04, 2.869824762166355e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
