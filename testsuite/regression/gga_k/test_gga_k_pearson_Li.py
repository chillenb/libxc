
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pearson_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.633280637773829e+01, 8.088622049944725e+00, 4.450097630235774e-01, 1.318494017711679e-01, 2.114298233719827e-02, 6.835105025615174e-04, 2.428649364816142e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pearson_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.596807913717939e+01, 2.601562798802092e+01, 1.240841489641492e+01, 1.242950646937285e+01, 7.641728179311282e-01, 7.633087013636337e-01, 2.138922213111689e-01, 1.037631566676464e-03, 3.873421245852490e-02, 1.043718048719901e-06, 1.147408072661821e-03, 1.130778374315752e-03, 4.656519441406371e-07, 2.353363432516427e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pearson_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.106730519317428e-03, 0.000000000000000e+00, 2.101232972047889e-03, 5.592469022901138e-03, 0.000000000000000e+00, 5.580877435166931e-03, -1.074328316818682e-02, 0.000000000000000e+00, -1.063766164383377e-02, 2.885210515119467e+00, 0.000000000000000e+00, -6.803837313183681e-07, -7.649848725049586e+00, 0.000000000000000e+00, -3.498463468859543e-10, -8.108283903978211e-07, 0.000000000000000e+00, -7.156976281136336e-07, -6.475245103495935e-11, 0.000000000000000e+00, -3.984163662264798e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
