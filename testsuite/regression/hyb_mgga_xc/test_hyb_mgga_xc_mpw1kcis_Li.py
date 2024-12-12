
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpw1kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.573914882152916e+00, -1.132166261364862e+00, -3.736014879061960e-01, -1.372573895703285e-01, -6.887053905439174e-02, -1.498804061889291e-03, -8.416297299886014e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpw1kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.990276110736924e+00, -1.991897312327085e+00, -1.369830794559288e+00, -1.370819747033980e+00, -3.348594298658411e-01, -3.346794169638458e-01, -1.899371906190659e-01, -1.126936621801362e-01, -6.760857626376160e-02, -4.460875679405444e-02, -5.662286177965169e-03, -5.374573352862407e-03, -3.333319187476136e-07, -2.366781241644666e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.189213298246671e-04, 6.147937556578381e-05, -1.182415138410872e-04, -6.118859049075507e-04, 2.026042608264738e-04, -6.092962471661314e-04, -8.032674761524584e-02, 1.042811294544413e-02, -8.022743328693174e-02, 1.521862089584591e+01, 4.862987484677894e+00, 3.151141466405566e+01, -3.773157704017504e+01, 2.591897797340521e+01, 4.048079347598562e+02, 2.967759944368812e+01, 1.042064888636250e+00, 2.791464834070979e+01, 4.576044847645651e+02, 2.643505997762938e+02, 6.313502205283251e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.456170580497749e-03, -6.467444957563024e-03, -4.877636520417854e-03, -4.890877574680303e-03, -1.047322136034373e-03, -1.101491606789539e-03, -6.904924272081766e-01, -2.443810795953947e-06, -5.840882687653810e-02, -1.572871022272044e-08, -1.148406634714330e-09, -2.527468106101991e-06, -3.203183121790221e-19, -3.695772064639163e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
