
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ft97_a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.788954322872422e+00, -1.278809428304224e+00, -4.695573551447962e-01, -1.597295904707497e-01, -8.421035053786528e-02, -1.416961711898083e-01, -5.549363015709891e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ft97_a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.244141855558072e+00, -2.246288073357224e+00, -1.511534212157127e+00, -1.512925381506603e+00, -2.873505244625332e-01, -2.873370406003735e-01, -2.055683641710115e-01, -4.128533228614715e-02, -6.295597732274144e-02, -8.315552532078314e-03, -4.237126599296179e-02, -4.252355722745642e-02, -7.999542707130304e-03, -6.905652033849541e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ft97_a_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.401079735437268e-04, 0.000000000000000e+00, -2.392505995824550e-04, -1.016129670161983e-03, 0.000000000000000e+00, -1.012722093023294e-03, -1.642108353550008e-01, 0.000000000000000e+00, -1.641229115549165e-01, -3.607170852088062e+00, 0.000000000000000e+00, -1.393095287709258e+03, -1.079334823070588e+02, 0.000000000000000e+00, -5.007637159893069e+07, -1.211398161259083e+03, 0.000000000000000e+00, -1.213330990609427e+03, -1.484815947297993e+08, 0.000000000000000e+00, -4.420304091649116e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
