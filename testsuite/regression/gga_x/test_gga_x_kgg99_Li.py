
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_kgg99_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kgg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.702472744126564e+00, -1.231805627450616e+00, -4.525334347080221e-01, -1.510601409775859e-01, -8.239044017867407e-02, -1.651360264079017e-01, 1.074742777375796e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_kgg99_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kgg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.071584721640917e+00, -2.073597006727032e+00, -1.390519148899903e+00, -1.391795890071115e+00, -3.058258837104519e-01, -3.057343507520521e-01, -1.905948139914814e-01, -4.256215565619695e-02, -6.533733336748536e-02, 2.322876876197901e-05, -4.329334041574659e-02, -4.362465501368100e-02, 1.551548315149997e-05, 1.103009096947592e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_kgg99_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kgg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.383427609613879e-04, 0.000000000000000e+00, -3.371845977211314e-04, -1.323732070609732e-03, 0.000000000000000e+00, -1.319519268801664e-03, -1.442556324671095e-01, 0.000000000000000e+00, -1.442060117189393e-01, -5.261598699934686e+00, 0.000000000000000e+00, -1.679220939047445e+03, -9.783006237165448e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.461877622219165e+03, 0.000000000000000e+00, -1.463456420264557e+03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
