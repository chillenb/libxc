
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_gea4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.637344387540874e+01, 8.129342956078471e+00, 1.076152436962529e+00, 1.911733286344236e-01, "nan", inf, inf]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_gea4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590338184731921e+01, 2.595002483101280e+01, 1.218442734410378e+01, 1.220593643779507e+01, -6.378265781957098e-01, -6.464629395321624e-01, 2.137410611268390e-01, -6.526767920188929e+02, "nan", "nan", "nan", "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.260988770260625e-03, 0.000000000000000e+00, 2.256536687059151e-03, 7.130318233677347e-03, 0.000000000000000e+00, 7.111286496341699e-03, 1.178346074987511e+00, 0.000000000000000e+00, 1.182606486559604e+00, 2.990089390398035e+00, 0.000000000000000e+00, 9.122046554156674e+06, 8.131771108256184e+01, 0.000000000000000e+00, "nan", 7.037181576390831e+06, 0.000000000000000e+00, -7.242851383424811e+172, 9.911536039745166e+14, 0.000000000000000e+00, -1.636085705863049e+274]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.653580145455081e-01, 1.653388242000677e-01, 1.639846420653969e-01, 1.639877854936166e-01, 1.302599493667405e-01, 1.301201710832171e-01, 1.658924924549134e-01, -2.438496609369523e+01, 1.490832850773634e-01, "nan", -2.185505888324307e+01, 2.338553018172759e+167, -2.538444204940310e+04, 1.586028629970346e+263]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
