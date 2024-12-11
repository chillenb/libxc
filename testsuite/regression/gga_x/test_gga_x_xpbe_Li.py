
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_xpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.797934105669159e+00, -1.288605648882415e+00, -4.290692950111002e-01, -1.602106230239748e-01, -8.210260991304365e-02, -2.185815240525332e-02, -4.084434955186899e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_xpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.237779756713953e+00, -2.239918149264107e+00, -1.512143742907454e+00, -1.513515049289662e+00, -3.950330516486435e-01, -3.952378596828543e-01, -2.050588333945555e-01, -2.778086612252677e-02, -7.487446324260588e-02, -8.827785181730317e-04, -2.920700305524158e-02, -2.899755082952703e-02, -5.896471317260719e-04, -4.191855480435271e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_xpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.715321874169816e-04, 0.000000000000000e+00, -2.705939430292365e-04, -1.081721289855161e-03, 0.000000000000000e+00, -1.078236232706011e-03, -8.586182738717617e-02, 0.000000000000000e+00, -8.566270880047508e-02, -4.192947027739189e+00, 0.000000000000000e+00, -3.434449405623788e-01, -7.570345020776475e+01, 0.000000000000000e+00, -2.197337371659084e+00, -3.490051108145836e-01, 0.000000000000000e+00, -3.259127267711940e-01, -1.599581106199252e+00, 0.000000000000000e+00, -2.289635959376285e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
