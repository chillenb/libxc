
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_cam_s12g_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.546033768676482e+00, -1.039435579476166e+00, -2.909605308679479e-01, -1.074630668752257e-01, -5.552192139373466e-02, -1.311272355332941e-02, -2.449329131270404e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_cam_s12g_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.073994392316706e+00, -2.076249978492830e+00, -1.289150597480168e+00, -1.290613470006872e+00, -3.043136075712833e-01, -3.044651409774941e-01, -1.429605282454243e-01, -1.667608612684115e-02, -5.198823548469271e-02, -5.293804086222960e-04, -1.753415555319407e-02, -1.740751943789571e-02, -3.535962377245842e-04, -2.513747106356408e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_cam_s12g_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.312952520874907e-05, 0.000000000000000e+00, -9.258167055921804e-05, -7.959148694968485e-04, 0.000000000000000e+00, -7.926774725514537e-04, -4.257482815341374e-02, 0.000000000000000e+00, -4.242471619794985e-02, -5.220212729412625e-01, 0.000000000000000e+00, -1.043041870067843e-01, -4.833188049119957e+01, 0.000000000000000e+00, -6.665497775711927e-01, -1.060070966398434e-01, 0.000000000000000e+00, -9.898681851312149e-02, -4.852229210587372e-01, 0.000000000000000e+00, -6.945464313393347e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
