
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_mod_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.794390282489741e+00, -1.283599990801528e+00, -4.160529752795550e-01, -1.600245936712595e-01, -8.050334070733987e-02, -2.054448767380057e-02, -3.838586978702006e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_mod_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.242610998917530e+00, -2.244744738656832e+00, -1.518990691262272e+00, -1.520359744361454e+00, -4.007657843631035e-01, -4.009425284468718e-01, -2.053102166657525e-01, -2.611573555879101e-02, -7.640105726297217e-02, -8.296433205696294e-04, -2.745724184444906e-02, -2.725994724704620e-02, -5.541555286587651e-04, -3.939542015820133e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_mod_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.551526109267178e-04, 0.000000000000000e+00, -2.542737671301495e-04, -1.010481535550560e-03, 0.000000000000000e+00, -1.007239291406087e-03, -7.467169197019846e-02, 0.000000000000000e+00, -7.448739492709580e-02, -3.950167678896367e+00, 0.000000000000000e+00, -2.777146430098337e-01, -6.769680846735837e+01, 0.000000000000000e+00, -1.776440715998272e+00, -2.822172314679048e-01, 0.000000000000000e+00, -2.635411218698511e-01, -1.293183348170954e+00, 0.000000000000000e+00, -1.851058914805758e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
