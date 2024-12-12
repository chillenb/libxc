
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_dldf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.321930964397142e-01, -4.753460511538797e-01, -2.931884103189796e-01, -3.839659615030400e-02, -4.374999762671004e-02, -3.645378442403428e-02, -6.837164200236529e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_dldf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.074122871619339e-01, -3.070840942248750e-01, -3.541182606428543e-01, -3.538382243765332e-01, -1.619054544507529e-02, -1.556864269464615e-02, -1.929561369277040e-02, -4.601761688034355e-02, -1.037146186815462e-02, -1.477697160989565e-03, -4.835943555700360e-02, -4.800190793026439e-02, -9.870368605846842e-04, -7.016958895317386e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_dldf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.262380464401676e-04, 0.000000000000000e+00, -1.256808185566011e-04, -6.491091387584414e-04, 0.000000000000000e+00, -6.466192402641352e-04, -1.555471924620996e-01, 0.000000000000000e+00, -1.551042911128400e-01, -1.553335670285249e+00, 0.000000000000000e+00, -3.472543350877104e+00, -8.676940137817112e+01, 0.000000000000000e+00, -2.237382438827538e+01, -3.527134781813097e+00, 0.000000000000000e+00, -3.293848947142306e+00, -1.628751063187726e+01, 0.000000000000000e+00, -2.331395341387186e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_dldf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.671548356727144e-02, -4.679932124040789e-02, -2.286273087511870e-02, -2.295815545605301e-02, -9.875525424156827e-03, -1.019730942870308e-02, -8.167572307021265e-01, -3.603399102479492e-06, -6.957225215816157e-02, -7.372626452648307e-10, -1.782356363858275e-09, -3.889490668173692e-06, -1.002745566907845e-20, -8.225315209185157e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
