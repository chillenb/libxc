
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.795621817795890e+00, -1.286840269641162e+00, -4.516234599031203e-01, -1.600634888252786e-01, -8.412448185373707e-02, -2.055687026651119e-02, -3.838588870220200e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.237909922048110e+00, -2.240055537526323e+00, -1.507220823647887e+00, -1.508604206751548e+00, -3.775195450638849e-01, -3.779566559446648e-01, -2.051588605826173e-01, -2.615912580860030e-02, -6.897000622535125e-02, -8.296468243503950e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -3.939545845223386e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.660258832714713e-04, 0.000000000000000e+00, -2.650902977128388e-04, -1.095265278010605e-03, 0.000000000000000e+00, -1.091656714166584e-03, -1.089249819989605e-01, 0.000000000000000e+00, -1.086119284281333e-01, -4.049055084293795e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.452287993327870e+01, 0.000000000000000e+00, 0.000000000000000e+00, -6.632227216413753e-309, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
