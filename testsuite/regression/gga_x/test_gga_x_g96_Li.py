
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_g96_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.810952711804909e+00, -1.293260668061384e+00, -4.388982593011232e-01, -1.616558242314840e-01, -8.081647078632306e-02, -7.416710129077113e-01, -2.896176971905065e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_g96_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.263931567792103e+00, -2.266039656717362e+00, -1.547865418262097e+00, -1.549234434814271e+00, -2.863688109334300e-01, -2.859606135258080e-01, -2.062602403550713e-01, 4.805213203007097e-01, -7.055983028130908e-02, 1.386810451164863e+00, 4.645292819541305e-01, 4.789425701208178e-01, 1.911751268653273e+00, 1.982150049758933e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_g96_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.564430664213053e-04, 0.000000000000000e+00, -2.556373883774949e-04, -9.261672797469372e-04, 0.000000000000000e+00, -9.232914272988185e-04, -1.448724396305061e-01, 0.000000000000000e+00, -1.449801848143169e-01, -4.388166070582834e+00, 0.000000000000000e+00, -1.426341515139263e+04, -8.139134101331642e+01, 0.000000000000000e+00, -3.165103141685486e+09, -1.198619527456992e+04, 0.000000000000000e+00, -1.219138339404304e+04, -1.249005731325484e+10, 0.000000000000000e+00, -4.312073264213905e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
