
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_lc_tmlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.837951051370580e+00, -1.249893512045301e+00, -2.078664056617846e-01, -5.597783951938470e-02, -5.622418239161213e-03, -2.131814233768707e-04, 7.108077144408422e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.413774530041078e+00, -2.415998747130998e+00, -1.601829548524921e+00, -1.603220853919812e+00, -3.167680209311069e-01, -3.184406238202552e-01, -9.625437854966037e-02, -8.222252883897752e-02, -9.778074876596175e-03, -3.029946517814971e-02, 1.777102203461716e-03, -2.770549555181160e-03, 2.972267841815829e-05, -9.360849208453858e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.483603137192532e-04, 5.222815421851711e-06, -2.475570231025202e-04, -1.008289906588437e-03, 3.646941789248587e-05, -1.005331088576178e-03, -5.922607399294519e-02, 4.773762863586187e-02, -5.824407433613568e-02, -6.089202309490228e-01, 4.596134769453040e+00, -2.735152374873195e+00, -3.650328389496979e+00, 2.356939734329661e+01, 1.006662589456961e+01, -1.275175087249358e+01, 7.936097321777658e-02, -6.063633201872509e+00, -1.621583122540737e+02, 0.000000000000000e+00, -7.621536236571052e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.000567614718288e-02, 9.999970639466666e-03, 1.223018725594228e-02, 1.222417603196342e-02, 1.167154973307958e-02, 1.164885261276568e-02, 2.711395046374277e-02, 2.486855527300091e-05, 3.865027705576260e-03, 9.452377746820062e-10, 2.862813438631864e-05, 2.798287213837665e-05, 2.818757779055703e-10, 1.013252626603096e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
