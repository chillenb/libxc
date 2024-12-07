
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pc07_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.637344040071861e+01, 8.129322970620798e+00, 3.215767055237889e+00, 1.329065774512231e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pc07_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590340898319974e+01, 2.595005324400479e+01, 1.218457349317040e+01, 1.220608228358678e+01, -3.210976564408325e+00, -3.220563663050524e+00, 2.137410943966468e-01, -3.048211395774484e+00, "nan", "nan", "nan", "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.260932041573275e-03, 0.000000000000000e+00, 2.256477963661392e-03, 7.129372593127306e-03, 0.000000000000000e+00, 7.110345750403334e-03, 4.153406735139716e+00, 0.000000000000000e+00, 4.158710224823614e+00, 2.990069963510749e+00, 0.000000000000000e+00, 7.830942109870746e+04, 4.181517806762772e+02, 0.000000000000000e+00, "nan", 6.734683730388351e+04, 0.000000000000000e+00, "nan", 8.236143116105227e+09, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.653586667210079e-01, 1.653395021347444e-01, 1.639883568126447e-01, 1.639914903897455e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.658926552568793e-01, 3.546182892022587e-03, 0.000000000000000e+00, "nan", 3.937699032231639e-03, "nan", 3.548803392260318e-06, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
