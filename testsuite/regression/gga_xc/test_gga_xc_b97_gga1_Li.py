
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_b97_gga1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.930746857391761e+00, -1.348596135927326e+00, -4.981047418529934e-01, -1.709605594217768e-01, -8.747627994608087e-02, -2.590000864172848e-02, -5.352494608673188e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_b97_gga1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.599989870958519e+00, -2.602293958278725e+00, -1.782955497079763e+00, -1.784478377213218e+00, -1.923870937253413e-01, -1.931887110686779e-01, -2.303814189436954e-01, 9.304522475837043e-01, -3.833776404019909e-02, 6.023655721727902e-01, -3.639052838030089e-02, -3.491576748179485e-02, -1.130959578574161e-03, 4.468885847061827e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_b97_gga1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.846656896440077e-05, 0.000000000000000e+00, 6.857666837950450e-05, 1.088290788678317e-06, 0.000000000000000e+00, 2.283565926785698e-06, -2.304262669802019e-01, 0.000000000000000e+00, -2.301018034150785e-01, 9.998910210207769e-01, 0.000000000000000e+00, 1.380539771666109e+02, -1.693435309709663e+02, 0.000000000000000e+00, 1.660392627695325e+04, -1.302931928980753e+00, 0.000000000000000e+00, -1.000549069855662e+00, -1.099986839112762e+01, 0.000000000000000e+00, 2.413500253222350e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
