
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.995244848932251e+00, -1.406981859010935e+00, -3.435407881411591e-01, -1.787868175582901e-01, -7.503569627035427e-02, -1.288097607114886e-02, -2.370364538993883e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.486236463209116e+00, -2.488663582594384e+00, -1.660037303048143e+00, -1.661502927203228e+00, -4.255507429473891e-01, -4.269741489909875e-01, -2.294071051513959e-01, -1.638541502943909e-02, -8.822409314052169e-02, -5.196768930812412e-04, -1.682454645373645e-02, -1.710504557726964e-02, -3.388209681340489e-04, -2.467665670758793e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.335220311268524e-04, 0.000000000000000e+00, -5.316930277474357e-04, -2.113824316056032e-03, 0.000000000000000e+00, -2.107725813612179e-03, -5.428824008061479e-02, 0.000000000000000e+00, -5.607911596645309e-02, -8.161242026639352e+00, 0.000000000000000e+00, -1.008898397647273e+01, -7.879136063785204e+01, 0.000000000000000e+00, -2.519683132502385e+04, -3.738628003677491e-01, 0.000000000000000e+00, -9.022337643408989e+00, -7.708158194869603e-01, 0.000000000000000e+00, -1.140693858577398e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_0_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.968581198560761e-02, 1.967093191101848e-02, 2.721516458299390e-02, 2.720903712645394e-02, 1.457450695052447e-02, 1.563408063756522e-02, 2.197305496195787e-01, 1.290662115014003e-04, 2.024862387193379e-01, 1.026612446432876e-05, 1.132702454939852e-07, 1.313195983856130e-04, 7.410729568756063e-16, 4.976074950818768e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
