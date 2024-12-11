
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
    ref_tgt = numpy.asarray([-6.531066012236973e+00, -2.246952286824099e+00, -3.911434595697234e-02, -9.646242372808770e-01, -1.769244501315983e-02, -1.562422520672424e-06, -5.156274692617661e-12])
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
    ref_tgt = numpy.asarray([-1.956272868938096e+01, -1.962121124100331e+01, -6.729569194384173e+00, -6.752115753309807e+00, -1.159403982235118e-01, -1.187474685504496e-01, -2.894835520536680e+00, -7.792869847501742e-06, -5.307734408225665e-02, -6.279792212216945e-10, -1.868233783496872e-07, -8.636881398872306e-06, -1.264304275378080e-15, -5.851928609572138e-11])
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
    ref_tgt = numpy.asarray([1.546156974095283e+00, 1.550841158396755e+00, 5.465574292768611e-01, 5.488233315072635e-01, 1.170140132350010e-02, 1.229054713851583e-02, 4.575354692030820e+01, 1.100152557229859e-06, 2.468971935425923e-01, 2.239435368427139e-10, 5.437614747478248e-10, 1.187874358583093e-06, 3.046027737256256e-21, 1.645297314713668e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
