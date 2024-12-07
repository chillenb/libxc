
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mn12_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.447270563509952e+00, -1.650051559798482e+00, -2.803359015971620e-01, -7.254474697161639e-02, -1.140022025358949e-01, -5.065080468324151e-02, -9.987905130668052e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mn12_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.411860833791614e+00, -1.405678652300048e+00, -2.186231226219066e+00, -2.187162580980473e+00, -5.282849729819781e-01, -5.363663739269172e-01, -2.447946223707027e-01, -6.331134159067488e-02, -8.339322013839484e-02, -2.156595436973280e-03, -6.641620039762842e-02, -6.586253285943519e-02, -1.441432686435681e-03, -1.025110713354982e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn12_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.475611225719283e-03, 0.000000000000000e+00, -4.469514457043474e-03, -7.235440744962645e-03, 0.000000000000000e+00, -7.236174239437957e-03, 8.120117950220671e-03, 0.000000000000000e+00, 9.583976964925548e-03, 4.744643390790914e+01, 0.000000000000000e+00, -3.985522051607108e+00, -2.059547699715242e+02, 0.000000000000000e+00, -2.716518627533277e+01, -4.044441608659973e+00, 0.000000000000000e+00, -3.770934312068575e+00, -1.978796809506917e+01, 0.000000000000000e+00, -2.833485228380944e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn12_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn12_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.199618153679090e-02, 9.129871917164645e-02, 1.903945733292817e-01, 1.907078681564633e-01, 4.074050580341968e-02, 4.172072397539885e-02, 1.176118387718578e+00, -1.387664712027105e-05, 2.688386126202418e-01, -2.907986186409703e-09, -6.848169787939840e-09, -1.496002199027692e-05, -3.955736157851438e-20, -3.245377147399184e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
