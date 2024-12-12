
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan01_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.565506159365746e-02, -2.438851412915902e-02, -1.479585590054271e-02, -2.017809035207221e-04, -3.770650406405979e-08, -1.067544432784334e-03, -5.788916855931726e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan01_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.481800607279643e-02, -3.474520562634345e-02, -3.777531514100179e-02, -3.770993227576814e-02, -4.754172278376294e-02, -4.756162747970717e-02, -1.604118121030944e-03, -1.591388012132206e-01, -1.301984949276472e-02, -9.352481170773772e-02, -2.008473861525348e-03, -2.023032292361654e-03, -1.075222045922506e-05, -1.347139200205775e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.684994138827990e-05, 3.369988277655980e-05, 1.684994138827990e-05, 7.401659021994576e-05, 1.480331804398915e-04, 7.401659021994576e-05, 1.726081552598375e-02, 3.452163105196750e-02, 1.726081552598375e-02, 2.122257625368660e+00, 4.244515250737320e+00, 2.122257625368660e+00, 7.597444024302327e+01, 1.519488804860465e+02, 7.597444024302327e+01, 2.992275437094825e+00, 5.984550874189649e+00, 2.992275437094825e+00, 6.472758373402569e+03, 1.294551674680514e+04, 6.472758373402569e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.564890439520451e-03, -1.564890439520450e-03, -2.241323003945220e-03, -2.241323003945220e-03, -7.230604895251820e-03, -7.230604895251814e-03, -8.058161637051341e-02, -8.058161637049560e-02, -1.816909583635969e-01, -1.816909582168075e-01, -1.455323796520347e-09, -1.455323796520348e-09, -1.665973269699082e-18, -1.665973269699082e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
