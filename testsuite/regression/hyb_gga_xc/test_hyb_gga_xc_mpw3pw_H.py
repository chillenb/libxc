
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw3pw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.335072208147243e-01, -4.876811137389594e-01, -3.021257702465058e-01, -1.152053481551450e-01, -1.654438175310947e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw3pw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.035599494032931e-01, 1.323296845606973e+00, -6.202297212489201e-01, 6.211397107975568e+01, -3.587063814330751e-01, 3.893339510396477e+01, -1.011287471923934e-01, 3.456077612987917e-01, -2.740865779708037e-03, -1.665983505669956e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw3pw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.484475377524626e-03, 2.875447401649912e-02, 1.437723700824955e-02, -1.095002963309565e-02, 1.497962924321718e-02, 7.489814621608586e-03, -8.774975006699917e-02, 6.672165099521191e-02, 3.336082549760595e-02, -5.781172414417069e+00, 1.628994012641055e-01, 8.144970063205273e-02, 5.181317358480746e+02, 1.630284078191381e-03, 8.151420390956904e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
