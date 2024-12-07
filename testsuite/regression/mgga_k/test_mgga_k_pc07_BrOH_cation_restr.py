
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pc07_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.179913268233657e+03, 2.179921940430797e+03, 2.179969889976884e+03, 2.179840421744003e+03, 2.179906934047591e+03, 2.179906934047591e+03, 5.807115976593943e+01, 5.807580257412963e+01, 5.818083268328116e+01, "nan", 5.807952410246949e+01, "nan", 2.195150665197498e+00, "nan", "nan", "nan", "nan", "nan", 6.132771423699879e-01, "nan", 2.661285023524655e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.276633091046214e+02, "nan", "nan", 2.720444925399673e+01, 2.696575746413651e+01, 2.837135809099123e+01, 2.815777963975927e+01, 2.651392869737744e+01, "nan", "nan", "nan", "nan", "nan", "nan", 1.704091056345955e+00, 9.219337397706971e-01, "nan", 8.996643050506318e-01, 1.904655420255587e+01, "nan", "nan", 6.678040803851272e-01, "nan", "nan", 8.107085906494392e-01, 4.134218398231095e-01, "nan", "nan", "nan", 1.583318566924559e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.335405365054589e+00, "nan", 1.064541222388073e+00, "nan", 6.698639965281545e-01, "nan", "nan", 9.995408978459779e-01, "nan", "nan", "nan", 7.143488347941436e-01, "nan", "nan", "nan", 4.583191396995908e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pc07_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.321447155285378e+03, 3.321469980474100e+03, 3.321570279523869e+03, 3.321230699303235e+03, 3.321408800089758e+03, 3.321408800089758e+03, 1.542676560217810e+02, 1.540250319020176e+02, 1.484577914257034e+02, "nan", 1.539841958657596e+02, "nan", -2.195150660357235e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", -2.426592480007680e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.997295027371406e+02, "nan", "nan", -2.720444925399673e+01, -2.696575746413649e+01, -2.837135809099123e+01, -2.815777963975924e+01, -2.651392869737742e+01, "nan", "nan", "nan", "nan", "nan", "nan", 2.580455735899354e+00, -9.219337397706977e-01, "nan", -8.996643050506312e-01, -3.174425700425978e+01, "nan", "nan", -6.669284906701317e-01, "nan", "nan", -8.107085906494390e-01, -4.122777827275363e-01, "nan", "nan", "nan", 2.587014034500667e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 2.009717247072718e+00, "nan", -1.064541222388072e+00, "nan", -6.698639965281546e-01, "nan", "nan", -9.995408895176420e-01, "nan", "nan", "nan", -7.143488347941434e-01, "nan", "nan", "nan", -4.582737641733465e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.892802525762136e-07, 7.892715496630191e-07, 7.892341852347175e-07, 7.893636694755251e-07, 7.892956291028284e-07, 7.892956291028284e-07, -9.199539569523910e-04, -9.161243043387094e-04, -8.278050915450610e-04, "nan", -9.152962497804252e-04, "nan", 2.087894143398975e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.274065925928007e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 5.310696839859488e-05, "nan", "nan", 8.405710951977128e-03, 8.225420308312626e-03, 8.540671341115710e-03, 8.377662191483195e-03, 8.242192632024158e-03, "nan", "nan", "nan", "nan", "nan", "nan", 3.650222691496617e-02, 8.282011699658828e+01, "nan", 1.046004417152880e+02, 0.000000000000000e+00, "nan", "nan", 3.249284044922582e+06, "nan", "nan", 3.437938582102904e+02, 2.043821369548235e+06, "nan", "nan", "nan", 3.534982976712733e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 5.317990130163296e-02, "nan", 5.986104049931646e+00, "nan", 1.499685834607407e+00, "nan", "nan", 6.763744307230733e-01, "nan", "nan", "nan", 1.665281126261038e+02, "nan", "nan", "nan", 9.043675061281249e+06, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.643542340413846e-01, 1.643543025163605e-01, 1.643544972430359e-01, 1.643534798665757e-01, 1.643540286085883e-01, 1.643540286085883e-01, 3.019759410947040e-01, 3.014985780159514e-01, 2.904954630053841e-01, "nan", 3.014036119236629e-01, "nan", 1.541486129806359e-10, "nan", "nan", "nan", "nan", "nan", 0.000000000000000e+00, "nan", 6.080315029990793e-03, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.649966269317527e-01, "nan", "nan", 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, "nan", "nan", "nan", "nan", "nan", "nan", 1.642082100139048e-01, 0.000000000000000e+00, "nan", 0.000000000000000e+00, 6.733927665856813e-67, "nan", "nan", 8.761498735157847e-04, "nan", "nan", 0.000000000000000e+00, 1.910114831456515e-03, "nan", "nan", "nan", 1.661200669552444e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.640637997317976e-01, "nan", 0.000000000000000e+00, "nan", 0.000000000000000e+00, "nan", "nan", 5.823776777973597e-10, "nan", "nan", "nan", 0.000000000000000e+00, "nan", "nan", "nan", -2.946585633451292e-58, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
