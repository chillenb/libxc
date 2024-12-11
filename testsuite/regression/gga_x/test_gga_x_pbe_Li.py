
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.794389906569591e+00, -1.283599506405938e+00, -4.160525866234336e-01, -1.600245731743457e-01, -8.050326257184416e-02, -2.054448759051530e-02, -3.838586978689267e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.242611416572563e+00, -2.244745156136328e+00, -1.518991128475153e+00, -1.520360181741196e+00, -4.007651747901766e-01, -4.009419173713364e-01, -2.053102412989374e-01, -2.611573526715777e-02, -7.640100912316396e-02, -8.296433205460150e-04, -2.745724150270820e-02, -2.725994692390995e-02, -5.541555286527651e-04, -3.939542015794342e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.551510359982956e-04, 0.000000000000000e+00, -2.542721973766889e-04, -1.010475829343740e-03, 0.000000000000000e+00, -1.007233602327459e-03, -7.467173619170629e-02, 0.000000000000000e+00, -7.448743999308474e-02, -3.950142390526672e+00, 0.000000000000000e+00, -2.777165088023333e-01, -6.769668583688618e+01, 0.000000000000000e+00, -1.776452680437340e+00, -2.822191269690022e-01, 0.000000000000000e+00, -2.635428921677539e-01, -1.293192057875573e+00, 0.000000000000000e+00, -1.851071381863682e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
