
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rge2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.770308510176618e+00, -1.253116663586781e+00, -4.124990636337771e-01, -1.587032286723242e-01, -7.723558515676374e-02, -2.055681730774229e-02, -3.838588870213389e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rge2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.268347872634656e+00, -2.270474137657007e+00, -1.543360125365463e+00, -1.544746132152450e+00, -3.331834160254594e-01, -3.333823979408724e-01, -2.068686232667230e-01, -2.615884186925238e-02, -6.809923201521983e-02, -8.296468242921040e-04, -2.750772838784266e-02, -2.730769673157839e-02, -5.541564195132548e-04, -3.939545845208714e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rge2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.560073731660037e-04, 0.000000000000000e+00, -1.554473416842723e-04, -6.679581689974628e-04, 0.000000000000000e+00, -6.656992640337407e-04, -1.051252670964239e-01, 0.000000000000000e+00, -1.049375890671866e-01, -2.334318195860684e+00, 0.000000000000000e+00, -2.181207982790535e-03, -7.632898673387312e+01, 0.000000000000000e+00, -3.548045411359179e-05, -2.470977217198933e-03, 0.000000000000000e+00, -2.197437154137547e-03, -9.831758246752769e-06, 0.000000000000000e+00, -8.509399423811510e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
