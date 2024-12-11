
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw91_mod_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.219274279023795e-01, -5.801389596411085e-01, -3.611353782475763e-01, -1.366201080517928e-01, -6.552418322909194e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw91_mod_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282595669943397e-01, -5.284899245882318e-17, -7.190825936484019e-01, -1.817127135732329e-16, -4.057885607366587e-01, -2.325763528756797e-17, -1.373359978325604e-01, -6.713731571365784e-17, -2.670520053336047e-04, -1.924686538054853e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw91_mod_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.498596150128613e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.528498229576451e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.653789753385021e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.045403815210956e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.435835118208536e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
