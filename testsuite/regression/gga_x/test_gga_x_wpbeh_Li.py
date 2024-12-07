
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_wpbeh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.789765778324625e+00, -1.283700191750667e+00, -4.166775936338946e-01, -1.596162519356472e-01, -8.028425505879203e-02, -2.055687022884794e-02, -3.838588863187333e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_wpbeh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.231381237351914e+00, -2.233551416360603e+00, -1.496075276768474e+00, -1.497445449519508e+00, -3.993664904204494e-01, -3.996550454022639e-01, -2.051548526356230e-01, -2.615912576067286e-02, -7.601985131204284e-02, -8.296468228303546e-04, -2.750809933077535e-02, -2.730803071836092e-02, -5.541564185036020e-04, -3.939545838005551e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_wpbeh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.638385798873760e-04, 0.000000000000000e+00, -2.628568631674682e-04, -1.131946520558437e-03, 0.000000000000000e+00, -1.128254332381582e-03, -7.575348637519275e-02, 0.000000000000000e+00, -7.551505080697071e-02, -3.760752283474782e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.789176185627373e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
