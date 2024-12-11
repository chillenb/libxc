
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wp04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wp04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.915225590894866e+00, -1.392610200255749e+00, -4.946008982042314e-01, -1.952441576242542e-01, -1.049060540231652e-01, -1.424955082152419e-01, -5.212032672559827e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wp04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wp04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.370042813224537e+00, -2.371956462134408e+00, -1.642408157612004e+00, -1.643596628200248e+00, -4.173844601591956e-01, -4.173061961655494e-01, -2.440476763152573e-01, -1.603704692796900e-01, -1.015131096554987e-01, -8.558382623801107e-02, -5.277575704814268e-02, -5.301705605994061e-02, -7.952758904143542e-03, -6.874963786761219e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wp04_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wp04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.633926868778012e-04, 5.222815421851711e-10, -2.625215043288850e-04, -9.907308917597294e-04, 3.646941789248588e-09, -9.876287709650931e-04, -1.102525585660347e-01, 4.773762863586187e-06, -1.102200985082270e-01, -4.239200388770010e+00, 4.596134769453040e-04, -1.287881287618864e+03, -7.363860626519931e+01, 2.356939734329661e-03, -4.663081594627344e+07, -1.119903042255670e+03, 7.936097321777658e-06, -1.121690168055491e+03, -1.384418150239012e+08, 0.000000000000000e+00, -4.124032230588313e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
