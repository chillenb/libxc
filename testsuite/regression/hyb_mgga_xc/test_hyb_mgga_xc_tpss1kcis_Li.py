
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss1kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.616671768301125e+00, -1.136017976875363e+00, -3.086375391749845e-01, -1.560469552406979e-01, -6.616455385626126e-02, -1.787708417967957e-02, -3.049908157094135e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss1kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.146796661438077e+00, -2.148452506623909e+00, -1.509091613861367e+00, -1.510153050424846e+00, -3.990876347144147e-01, -3.988338402012314e-01, -2.058420364251691e-01, -1.354725334850477e-01, -8.509459414734868e-02, -6.864732284984039e-02, -2.393765973972187e-02, -2.370204546315317e-02, -4.821160855014665e-04, -1.966226429802495e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.735248422000747e-05, 3.579307310313267e-04, 9.028884232086827e-05, 5.273450302663307e-04, 1.584861242149854e-03, 5.282416656105917e-04, 2.545806015628383e-01, 5.181750224872892e-01, 2.547639551625757e-01, 2.200778044402627e+01, 1.240351293287249e+01, 5.960417247679411e+00, 2.843071512027614e+02, 4.973769242805170e+02, 2.471494908932158e+02, 7.376559223333140e-04, 1.681208480457206e-03, -2.282078303870591e-01, 6.385048366306947e-09, 1.291118581827277e-08, -1.354925638539486e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.006420718878976e-03, 2.005190799950485e-03, 2.056806303813541e-03, 2.060464295810645e-03, -5.529318375785876e-04, -5.851696141679873e-04, 2.334468346757446e-02, -2.443752375539020e-06, -1.354250994820622e-02, -1.572871019254175e-08, -1.151970477230079e-09, -2.527401141795257e-06, -3.203490006160367e-19, -4.780721405173881e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
