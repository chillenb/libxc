
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revscan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.485105048062696e+00, -9.830550275060947e-01, -1.706608994594492e-01, -1.359342300974178e-01, -3.825676405147014e-02, -3.612441797238261e-03, -2.566332187339385e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revscan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.068136317974340e+00, -2.070047300769183e+00, -1.507634643349735e+00, -1.508407578813457e+00, -2.371400062040623e-01, -2.375118633637022e-01, -1.854714571763873e-01, 1.287661275094536e+00, -5.684578371897049e-02, 4.629547791041342e+00, 2.807054832064843e+01, 1.278942556090280e+00, 4.591551925280289e+04, -1.247410095397208e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revscan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.679013589121271e-04, 0.000000000000000e+00, -2.668645343934291e-04, -1.930286023808355e-03, 0.000000000000000e+00, -1.919266467104514e-03, -3.449937357022455e-01, 0.000000000000000e+00, -3.454056990018449e-01, -3.286605612222590e+00, 0.000000000000000e+00, -3.317481859583368e+04, -1.595464140323712e+02, 0.000000000000000e+00, -9.389006894072609e+09, -1.274875724682372e+04, 0.000000000000000e+00, -2.822736288751075e+04, -2.111083349168303e+09, 0.000000000000000e+00, -3.500974693710968e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revscan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.267490772853097e-02, 1.266047942700099e-02, 2.889467134682752e-02, 2.880992011897040e-02, 1.751075343222700e-03, 1.847590625062284e-03, 1.185655950160084e-01, 4.239675004846063e-01, 4.887097868039182e-02, 3.825422051231691e+00, 1.893374151193330e-01, 4.104035726836691e-01, 2.563194106399923e-01, 2.059081681678984e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
