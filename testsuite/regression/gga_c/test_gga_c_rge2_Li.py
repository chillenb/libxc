
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_rge2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.445850054416713e-02, -4.825401974549659e-02, -4.372801159324598e-03, -1.571461860062609e-02, -1.850705096717208e-03, -1.205760900065828e-08, -2.867443695681038e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_rge2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.170509133212619e-01, -1.169188567130123e-01, -1.042325149079505e-01, -1.041309861264517e-01, -2.152476612450284e-02, -2.153254214117802e-02, -2.373094882077849e-02, -1.032537815295125e-01, -8.452531727371881e-03, 4.218952061243852e-01, -7.802668386653795e-08, -7.841865755702272e-08, -1.802791043618163e-15, -2.134394278072854e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_rge2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.255282280779931e-05, 8.510564561559863e-05, 4.255282280779931e-05, 1.425296611777489e-04, 2.850593223554978e-04, 1.425296611777489e-04, 4.166624086776050e-03, 8.333248173552105e-03, 4.166624086776050e-03, 2.805371774207139e+00, 5.610743548414278e+00, 2.805371774207139e+00, 1.418823871261980e+01, 2.837647742523959e+01, 1.418823871261980e+01, 2.659344171297619e-04, 5.318688342673471e-04, 2.659344171297619e-04, 2.546558364230662e-06, 5.092949805140015e-06, 2.546558364230662e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
