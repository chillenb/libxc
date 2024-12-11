
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_gg99_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.790163803758837e+00, -1.292687369232369e+00, -4.671498456200610e-01, -1.589850756464425e-01, -8.553304523676741e-02, -1.657115855154629e-01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_gg99_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.188453412435599e+00, -2.190570999362880e+00, -1.471660229220748e+00, -1.473006033032583e+00, -3.253185737936536e-01, -3.252187511451071e-01, -2.011646654451607e-01, -4.329456885479110e-02, -6.952747415485917e-02, -2.489376768265149e-17, -4.406352269180672e-02, -4.438923569228342e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_gg99_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gg99", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.383427609613879e-04, 0.000000000000000e+00, -3.371845977211314e-04, -1.323732070609732e-03, 0.000000000000000e+00, -1.319519268801664e-03, -1.442556324671095e-01, 0.000000000000000e+00, -1.442060117189393e-01, -5.261598699934686e+00, 0.000000000000000e+00, -1.679220939047445e+03, -9.783006237165448e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.461877622219165e+03, 0.000000000000000e+00, -1.463456420264557e+03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
