
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sfat_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.596997349727755e+00, -1.084835186552295e+00, -2.045316849094132e-01, -5.483839151260785e-02, -7.515015525352277e-03, -5.953109113019369e-05, -4.091447092939645e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sfat_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.056551041243770e+00, -2.058637247101085e+00, -1.354476904809847e+00, -1.355804180712084e+00, -2.419210591085729e-01, -2.418522254580709e-01, -8.784424699483788e-02, -1.034196338367942e-04, -1.331634248404769e-02, -3.305774532712539e-09, -1.202218678311998e-04, -1.176265741233385e-04, -9.851233641049701e-10, -3.539419196525317e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sfat_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.286603798860511e-04, 0.000000000000000e+00, -2.279408050376506e-04, -7.946656911030970e-04, 0.000000000000000e+00, -7.923503619238215e-04, -3.655286875020381e-02, 0.000000000000000e+00, -3.651113967978286e-02, -8.303677057088461e-01, 0.000000000000000e+00, -5.600443050451844e-04, -1.471890100425194e+00, 0.000000000000000e+00, -1.244143764275257e-07, -6.488917765559035e-04, 0.000000000000000e+00, -6.072849293952049e-04, -3.096226960357199e-08, 0.000000000000000e+00, -1.497839873325033e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
