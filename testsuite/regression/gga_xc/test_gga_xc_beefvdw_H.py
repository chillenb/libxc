
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_beefvdw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.751428613983055e-01, -6.185019960213308e-01, -3.900434929410682e-01, -1.555484171804739e-01, -8.609630249572399e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_beefvdw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.929507413333772e-01, 4.808581348129403e-01, -7.784776752948900e-01, 2.938717684953902e+01, -4.232814128721532e-01, 1.643401247952471e+01, -1.755103951536400e-01, 8.093721243510245e-02, -1.142005915717980e-02, -4.243940879855832e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_beefvdw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.028182423809565e-03, 1.314675235597871e-02, 6.573376177989356e-03, -1.870679259428477e-02, 8.162131636810864e-03, 4.081065818405432e-03, -2.028383865344780e-01, 3.365556406905235e-02, 1.682778203452617e-02, -3.453170027368336e+00, 7.045917520327304e-02, 3.522958760163634e-02, -3.674432568145078e+00, 7.938502461169235e-04, 3.969251232395268e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
