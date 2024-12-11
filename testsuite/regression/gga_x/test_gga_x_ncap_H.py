
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ncap_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220374192371277e-01, -5.782486624084801e-01, -3.612844043931057e-01, -1.565625871974314e-01, -3.188556374024206e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ncap_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282753097898484e-01, -4.870211217843056e-17, -7.170896309961485e-01, -1.449243568499006e-16, -3.977483828317837e-01, -1.130626585050488e-17, -8.548901940340253e-02, -7.292855277263634e-17, 1.548936320031861e-01, -6.323503035619885e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ncap_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.699398589961639e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.503999467732608e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.833721408142769e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.387426332853514e+01, 0.000000000000000e+00, 0.000000000000000e+00, -4.634929312058839e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
