
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scanh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.816990437844009e+00, -1.235108041229435e+00, -2.758436994624706e-01, -1.651227096463890e-01, -6.239093062771038e-02, -4.547386790508292e-03, -9.455582452890025e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scanh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.495212767943486e+00, -2.497348093078269e+00, -1.760606786811037e+00, -1.761750859156837e+00, -3.610745955628776e-01, -3.615798278802710e-01, -2.209381471337786e-01, 1.219036098450826e-01, -8.174053273044234e-02, -5.415471788275213e-02, 1.644313638216742e-01, 2.899204510675554e-01, 9.134839913664121e-05, -1.610477432387999e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scanh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.452982764586134e-04, 1.420514145497902e-04, -2.440793214958277e-04, -1.055574739610774e-03, 1.115362495104119e-03, -1.049666595968882e-03, 1.919216543193820e-01, 4.180228328436082e-01, 1.914229071173094e-01, -6.473975353520993e-01, 7.035114068157315e+00, -7.378145592178359e+03, 1.057423061427924e+02, 2.641353630163130e+02, -6.378882663183207e+06, -7.698621850278327e+01, 8.575971354726938e-01, -6.599069122204098e+03, -4.323160390298982e+00, 4.666265396414997e-04, 2.333132698207498e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scanh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scanh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.210956013793018e-02, 1.209063488715444e-02, 1.877179252107553e-02, 1.874101509305964e-02, 8.615050174184940e-04, 9.784833375328357e-04, 3.871496943859193e-02, -2.633085183170653e-02, 5.370782624462382e-03, -4.651788011212272e-02, 1.141766140384773e-03, 9.620476078143811e-02, 5.270440913400262e-10, -6.324714924461792e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
