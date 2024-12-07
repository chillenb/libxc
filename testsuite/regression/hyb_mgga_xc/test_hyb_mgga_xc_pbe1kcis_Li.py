
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pbe1kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.440852005955473e+00, -1.035571910815716e+00, -3.351909717411675e-01, -1.255750986017700e-01, -6.279255012133005e-02, -1.606824015768978e-02, -2.994381869366008e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pbe1kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.833725528731599e+00, -1.835209963588766e+00, -1.258204707718398e+00, -1.259106593596772e+00, -3.422593375346861e-01, -3.422306010879668e-01, -1.758001615716369e-01, -1.283178608621129e-01, -6.377831995042001e-02, -4.525506479395076e-02, -2.158909507493723e-02, -2.141586398048631e-02, -4.323384843108595e-04, -3.074121556209420e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.328514559478440e-05, 6.147937556578381e-05, -8.269376620898828e-05, -5.217104056641415e-04, 2.026042608264738e-04, -5.193326873614403e-04, -5.191724511426259e-02, 1.042811294544413e-02, -5.174917622751610e-02, 1.592453161874132e+01, 4.862987484677894e+00, 2.215439745351850e+00, -2.837965985784797e+01, 2.591897797340521e+01, 1.158043805329873e+01, 3.009134705898724e-01, 1.042064888636250e+00, 3.159954387539928e-01, 1.311666101149212e+02, 2.643505997762938e+02, 1.307382980078092e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.456170580497749e-03, -6.467444957563024e-03, -4.877636520417854e-03, -4.890877574680303e-03, -1.047322136034373e-03, -1.101491606789539e-03, -6.904924272081766e-01, -2.443810795953947e-06, -5.840882687653810e-02, -1.572871022272044e-08, -1.148406634714330e-09, -2.527468106101991e-06, -3.203183121790221e-19, -3.695772064639163e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
