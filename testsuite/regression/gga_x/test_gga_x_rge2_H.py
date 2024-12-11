
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rge2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.218548373312290e-01, -5.692233038010985e-01, -3.474182308242710e-01, -1.373571094044007e-01, -7.399209449213098e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rge2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.285209600283193e-01, -1.118846798413020e-16, -7.274133571669349e-01, -1.956014163889531e-16, -4.068088970240634e-01, 3.836505016800696e-17, -1.250019964919550e-01, -4.247072858278165e-17, -9.865578161376747e-03, -9.845491034405753e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rge2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.491939847250049e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.465514376739502e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.232075100996173e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.544327549625047e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.761037149264747e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
