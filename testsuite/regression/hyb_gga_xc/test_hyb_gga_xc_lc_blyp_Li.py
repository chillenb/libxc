
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_blyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.668591270475402e+00, -1.145791218003628e+00, -1.703915532512297e-01, -4.361390439187689e-02, -4.022859160164044e-03, -2.104671731837936e-03, -3.003789033632164e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_blyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.139289387935025e+00, -2.141216562996795e+00, -1.434323093549479e+00, -1.435490285163836e+00, -3.481104111333925e-01, -3.484658713761274e-01, -7.562996854671362e-02, -8.227492255220456e-02, -7.749012495410590e-03, -3.029946575057469e-02, -2.740950096380236e-03, -2.830869016118783e-03, -2.079729998899352e-05, -9.360855102071484e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_blyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.247846681755698e-04, 5.222815421851711e-06, -2.241807389966481e-04, -7.616182018185221e-04, 3.646941789248587e-05, -7.599454645171853e-04, 7.199822855602615e-03, 4.773762863586187e-02, 7.403063122409535e-03, -4.417676834453706e-01, 4.596134769453040e+00, 3.448492092767159e+00, -2.548381681938448e-01, 2.356939734329661e+01, 1.767704994115178e+01, 4.080505562835862e-02, 7.936097321777658e-02, 4.101912202220664e-02, -3.058001984680710e-09, 0.000000000000000e+00, -1.479347944172945e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
