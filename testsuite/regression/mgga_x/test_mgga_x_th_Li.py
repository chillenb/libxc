
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_th_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.222382875450940e+00, -2.829112012640193e+00, -4.911580760911617e-02, -1.202693376202035e+00, -2.227937430613189e-02, -1.555606481490774e-06, -6.492931509265975e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_th_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.293665218038650e+01, -2.301253170074758e+01, -7.891168812129565e+00, -7.919148105733725e+00, -1.354226558489538e-01, -1.392717223721678e-01, -3.371132212879077e+00, -7.792869847672342e-06, -6.225120423114965e-02, -6.279792189919634e-10, -1.598545051485899e-07, -8.636881398872307e-06, -1.077987225922991e-15, -6.863372832086028e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_th_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.719623038265749e-03, 0.000000000000000e+00, -7.727756863341343e-03, -8.159574602989914e-03, 0.000000000000000e+00, -8.172516634097160e-03, -1.260017605456247e-02, 0.000000000000000e+00, -1.325147290279099e-02, -3.090097538539546e+02, 0.000000000000000e+00, -1.470680745444058e-02, -2.676605584691382e+01, 0.000000000000000e+00, -9.384128386114245e-02, -6.251177839638299e-06, 0.000000000000000e+00, -1.395836436580858e-02, -4.283235186196766e-12, 0.000000000000000e+00, -9.778250945245275e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_th_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.345750578638349e+00, 2.354981017911689e+00, 8.295833775591291e-01, 8.333983922888073e-01, 1.760020067185927e-02, 1.866342343199711e-02, 6.833748604894855e+01, 1.100152557229859e-06, 3.749179355678747e-01, 2.239435347931514e-10, 3.618836449380281e-10, 1.187874358583093e-06, 2.005928940962307e-21, 2.498414248110314e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
