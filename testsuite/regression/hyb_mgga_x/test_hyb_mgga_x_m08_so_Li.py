
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m08_so_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.485835522818679e-01, -4.234886489238622e-01, -2.430185398822894e-01, -5.427669269585984e-02, -5.003096189589213e-02, -1.354883458740378e-02, -2.523257011915360e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m08_so_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.371375731813278e-01, -6.440812398232868e-01, -1.803245558512586e-01, -1.774551097002170e-01, -3.112628527702509e-01, -3.125456321975519e-01, -1.003278855422354e-01, -1.733783463405891e-02, -4.594815601490161e-02, -5.453705645881326e-04, -1.819271939636505e-02, -1.810829025648993e-02, -3.642703081336502e-04, -2.589628246362770e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_so_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.041692198338238e-03, 0.000000000000000e+00, -1.039570583904807e-03, -1.087252220912482e-03, 0.000000000000000e+00, -1.094009642223790e-03, -2.492837377475031e-02, 0.000000000000000e+00, -2.487516093964106e-02, -1.655643690673287e+01, 0.000000000000000e+00, 5.870644409002859e-01, -5.591251236307487e+01, 0.000000000000000e+00, 3.821387908558923e+00, 6.068771171741366e-01, 0.000000000000000e+00, 5.564979442535318e-01, 2.781952532005215e+00, 0.000000000000000e+00, 3.982038301222058e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_so_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.207452472402367e-02, 9.299622373025594e-02, -2.601473238054856e-02, -2.617417223005582e-02, 7.083686884449848e-03, 7.364105920453437e-03, 1.762914546884494e+00, 5.460765513200752e-06, 4.012026374762526e-02, 1.147231646021823e-09, 2.712097696350638e-09, 5.884569869615716e-06, 1.552052385153334e-20, 1.279982209166900e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
