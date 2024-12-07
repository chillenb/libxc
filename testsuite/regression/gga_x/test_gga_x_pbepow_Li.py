
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbepow_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbepow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.796926167107225e+00, -1.290468594455765e+00, -4.465868300587331e-01, -1.601037247982297e-01, -8.588137267458233e-02, -2.055061142898511e-02, -3.838587912306229e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbepow_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbepow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.232735658577486e+00, -2.234895235573318e+00, -1.493020201721337e+00, -1.494423495633903e+00, -4.434647759773491e-01, -4.438440730767789e-01, -2.049987595271397e-01, -2.613718363011415e-02, -7.486501727613304e-02, -8.296450600378438e-04, -2.748237701140453e-02, -2.728371318618403e-02, -5.541559657459855e-04, -3.939544285007911e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbepow_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbepow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.778767444784239e-04, 0.000000000000000e+00, -2.768774987945459e-04, -1.195584212860175e-03, 0.000000000000000e+00, -1.191525466517884e-03, -7.371294815613280e-02, 0.000000000000000e+00, -7.342187747929715e-02, -4.153071342387767e+00, 0.000000000000000e+00, -1.404769662836865e-01, -8.674920568320516e+01, 0.000000000000000e+00, -8.913924747164431e-01, -1.427813714222427e-01, 0.000000000000000e+00, -1.333209965428284e-01, -6.614791624446168e-01, 0.000000000000000e+00, -7.136799454100227e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
