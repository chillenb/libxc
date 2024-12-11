
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_dk87_r1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.218434141562448e-01, -5.788512596755550e-01, -3.625489735974312e-01, -1.392393637531536e-01, -6.448828438064839e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_dk87_r1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.284615991674885e-01, -2.190157250821180e-17, -7.116015399794871e-01, -2.007582284800481e-16, -4.029891077217208e-01, 3.722975833586085e-18, -1.131899694969818e-01, -6.812647676558068e-17, -5.631449848910854e-03, -7.947305229775143e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_dk87_r1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.016883595441307e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.796235812011513e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.756090592668059e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.156377374602975e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.420827736017181e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
