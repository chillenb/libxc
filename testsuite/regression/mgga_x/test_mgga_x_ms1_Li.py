
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.969319084412495e+00, -1.377659415522470e+00, -3.638068604514504e-01, -1.773888116691506e-01, -7.535122459514798e-02, -1.599324568792378e-02, -2.987459668414639e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.583910972566227e+00, -2.586314369343599e+00, -1.782517810995451e+00, -1.784351998293234e+00, -3.578866090571458e-01, -4.320511799320523e-01, -2.342662754820935e-01, -2.033940779624498e-02, -9.061193572251158e-02, -6.456877785365629e-04, -2.138588305199891e-02, -2.123146167292991e-02, -4.312831992700227e-04, -3.066030134684825e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.983610524936618e-05, 0.000000000000000e+00, -6.939752103943549e-05, -2.849359675777044e-04, 0.000000000000000e+00, -2.798964138213262e-04, -1.223479255904295e-01, 0.000000000000000e+00, -2.599661295747773e-02, -1.247253741325321e+00, 0.000000000000000e+00, -1.245901098316482e-01, -2.156822910038924e+01, 0.000000000000000e+00, -1.417133609068484e+00, -1.268809106479767e-01, 0.000000000000000e+00, -1.182261644172310e-01, -5.805817216896689e-01, 0.000000000000000e+00, -1.798052025570625e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.019037451817267e-05, 9.878421832708159e-12, 7.166693911558697e-05, 1.431176505149434e-17, 2.283584003183302e-02, 7.085661373014425e-11, 6.806152970009213e-03, 7.796729907858791e-18, 6.629021296972384e-07, 2.524442521803118e-10, 2.944918714038297e-21, 2.778204598835961e-18, -9.510769382699616e-39, 4.218405656371013e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
