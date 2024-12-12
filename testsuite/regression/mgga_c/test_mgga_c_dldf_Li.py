
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_dldf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.232782153521985e-02, -6.500919473382578e-02, -9.923609661327567e-03, -2.504751731524652e-04, 5.026842465739646e-08, 1.260231527798556e-02, 1.921855143193928e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_dldf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.209504667242856e-02, -5.202110494847142e-02, -4.268741929311781e-02, -4.260246293109193e-02, -1.510056973960270e-01, -1.517884779981908e-01, -9.437830193786143e-03, 4.901009877444804e-01, -4.863969183937528e-03, 2.999566828292200e-01, 1.440453961684139e-02, 1.544229745138407e-02, 7.990214847310208e-05, 6.941819449287656e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.707669558533275e-05, 0.000000000000000e+00, 1.739515737424247e-05, -6.539059061161617e-05, 0.000000000000000e+00, -6.420464403058168e-05, 7.755352025236395e-02, 0.000000000000000e+00, 7.827219025399657e-02, 1.113661836632259e+01, 0.000000000000000e+00, 1.465574638377944e+02, 2.838272598368647e+01, 0.000000000000000e+00, 2.310615328065247e+05, 2.107119642442154e+00, 0.000000000000000e+00, 4.936028641646526e+01, 5.169473400443827e+00, 0.000000000000000e+00, 1.035711439445624e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.734653806928235e-03, -3.743910607996075e-03, -2.443533005139497e-03, -2.449936723516292e-03, -3.704686200568002e-03, -3.909604273474375e-03, -3.955242249987734e-01, -6.956828268041400e-04, -6.787660236213584e-02, -8.971988568629111e-05, -3.045500824271357e-07, -6.999434029559420e-04, -3.278229564960057e-15, -4.517978038815421e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
