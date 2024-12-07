
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b94_hyb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.277844423089405e+00, -1.073866371020148e+00, -4.174266304703520e-01, -1.244127334388073e-01, -7.329045699456353e-02, -1.433071115604445e-01, -4.184977659415168e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b94_hyb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.704372121445245e+00, -1.705644039550618e+00, -1.517637540934167e+00, -1.518968146344748e+00, -3.695385223735126e-01, -3.691621483887467e-01, -1.904392386103478e-01, -1.307320341474156e-01, -7.219055880811986e-02, -5.193769028647542e-02, -7.177077902309401e-02, -6.440112490704115e-02, -2.820452107542482e-02, -1.296245254457233e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.366644305792072e-05, 0.000000000000000e+00, 5.342497003604843e-05, -7.927175888179031e-04, 0.000000000000000e+00, -7.915112655551740e-04, -7.270391376374719e-02, 0.000000000000000e+00, -7.286308942244324e-02, -4.462471999128066e-01, 0.000000000000000e+00, -5.460891672226101e+02, -5.029936118790940e+01, 0.000000000000000e+00, -1.086109856256919e+07, -6.592477400419472e+02, 0.000000000000000e+00, -7.123232725722615e+02, -5.492406458010598e+07, 0.000000000000000e+00, -3.730767491278416e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-6.306605457201939e-06, -6.333611629489161e-06, -3.894275661348797e-03, -3.896793861919372e-03, -4.853477887565829e-03, -4.856152295362309e-03, -2.086298071304523e-02, -1.746713249095110e-03, -4.033105972734464e-02, -1.106320245256243e-03, -2.451237502724894e-03, -2.590845053434569e-03, -1.667173000460590e-03, -4.068704908524352e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b94_hyb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b94_hyb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.784694467328564e-03, -2.779664829017261e-03, 1.376642124935508e-02, 1.378060006417620e-02, 1.750464580043148e-02, 1.752059784966951e-02, 1.713019706196546e-02, 6.974486706566769e-03, 1.202897213699778e-01, 4.425196194835597e-03, 9.790602385573400e-03, 1.034968885679930e-02, 6.668663209614569e-03, 1.627480036208738e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
