
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan50_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.044360190518717e+00, -7.309232678955685e-01, -1.778298487209903e-01, -9.222035615247959e-02, -3.592031932800471e-02, -4.036179284911913e-03, -1.300059166887747e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan50_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.369757555088730e+00, -1.370927122139907e+00, -9.516754888431448e-01, -9.524454768236179e-01, -1.772957426712158e-01, -2.025060804939483e-01, -1.231235687085263e-01, -4.099623718840012e-03, -5.424834874962389e-02, -9.223664189898441e-02, -6.768828277463513e-03, 1.597888600172837e-01, -3.421601344204171e-05, -1.347139219250734e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.310352719861397e-05, 3.376676096858215e-05, -9.270965664095825e-05, -3.547055628019743e-04, 1.486544711016959e-04, -3.528254362530963e-04, -1.111747401016045e-01, 3.686155290429969e-02, -7.684418573975536e-02, 3.515363881272013e-01, 4.249088087620090e+00, -4.098799427740878e+03, 3.107159710339857e+01, 1.567412608661205e+02, -3.543945970216074e+06, 1.320559103728715e+01, 5.982918252051843e+00, -3.663396274636104e+03, 2.283086262214350e+04, 1.294551542135600e+04, 6.472757710678000e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([4.326750284810913e-03, 4.321629914421516e-03, 5.594951328404171e-03, 5.581857253613339e-03, 2.481235227177544e-02, 1.658959662675363e-02, -1.151071786520383e-02, -2.812670765339881e-02, -6.568412310704520e-02, -1.859720136019659e-01, -1.195352039542191e-10, 5.345377420457115e-02, -1.051159073018728e-19, -1.666241956151310e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
