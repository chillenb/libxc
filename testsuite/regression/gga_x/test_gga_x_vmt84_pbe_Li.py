
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt84_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794886727759138e+00, -1.284874947665236e+00, -4.308220838761798e-01, -1.600371169640247e-01, -8.190043907451384e-02, -9.314659151975629e-03, -6.423415171602323e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt84_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.240773187302569e+00, -2.242911497392815e+00, -1.514454846234930e+00, -1.515829316613708e+00, -3.836038514376462e-01, -3.838080365678115e-01, -2.052502689514728e-01, -1.669730984625479e-02, -7.340208104578751e-02, -1.189855392298189e-09, -1.738863030206981e-02, -1.733780728520939e-02, -3.025278567093242e-10, -1.300428873720614e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt84_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.594376078639808e-04, 0.000000000000000e+00, -2.585366131723065e-04, -1.043339497914471e-03, 0.000000000000000e+00, -1.039956116490044e-03, -9.253318157025441e-02, 0.000000000000000e+00, -9.233528962562508e-02, -3.989811667473679e+00, 0.000000000000000e+00, 4.850579763126110e+01, -7.833550300034086e+01, 0.000000000000000e+00, 6.032654285884784e-01, 4.050938052632484e+01, 0.000000000000000e+00, 4.133206640552479e+01, 4.391532728352826e-01, 0.000000000000000e+00, 6.286021305296537e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
