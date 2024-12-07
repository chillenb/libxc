
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wc04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.819812059250423e-01, -1.907796233847767e-01, -2.107755639393529e-01, -3.851636485216577e-02, -4.343307065678834e-02, -1.357923310204799e-01, -5.396353378617958e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wc04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.393984362842123e-02, -5.377045151476949e-02, -3.312132002545662e-02, -3.294246372813399e-02, -2.975903357282617e-02, -2.984634203030218e-02, -3.466163699824339e-02, -1.466935654851719e-01, -1.820149376991894e-02, -8.540370320034091e-02, -3.835496786612012e-02, -3.871819585954352e-02, -7.921069399452155e-03, -6.896665800806313e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wc04_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.739404753740154e-04, 5.222815421851711e-10, -2.730344051552281e-04, -1.030405632123752e-03, 3.646941789248588e-09, -1.027179281903715e-03, -1.146678453938203e-01, 4.773762863586187e-06, -1.146340860824175e-01, -4.408962416388778e+00, 4.596134769453040e-04, -1.339455494868951e+03, -7.658752070374577e+01, 2.356939734329661e-03, -4.849818271764808e+07, -1.164750418253371e+03, 7.936097321777658e-06, -1.166609110876680e+03, -1.439858236346981e+08, 0.000000000000000e+00, -4.289182262705693e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
