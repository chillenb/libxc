
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.732507745212576e+00, -1.221730388948232e+00, -3.544879013313179e-01, -1.056839108652973e-01, -2.969226723960760e-02, -3.386237689346041e-04, -1.636588711582796e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.180985544357427e+00, -2.183119265834357e+00, -1.457510782405371e+00, -1.458879097568206e+00, -3.390056065495266e-01, -3.391781557701281e-01, -1.506219438440546e-01, -6.360281265032873e-04, -2.871225707373489e-02, -1.322334335422265e-08, -7.716717397209627e-04, -7.503481762802904e-04, -3.940526617842013e-09, -1.415772181027223e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.545274389929117e-04, 0.000000000000000e+00, -2.536502264006382e-04, -1.007844475895852e-03, 0.000000000000000e+00, -1.004614243492031e-03, -7.460141429051365e-02, 0.000000000000000e+00, -7.441951990285478e-02, -2.712068011561597e+00, 0.000000000000000e+00, -7.898721212289796e-03, -4.818937022450574e+01, 0.000000000000000e+00, -1.901464088683739e-09, -9.952131705608035e-03, 0.000000000000000e+00, -9.012880629250460e-03, -2.652336587012084e-10, 0.000000000000000e+00, -9.547110283538582e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
