
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_otpss_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.869750463880407e+00, -1.314106061392590e+00, -3.400337038569259e-01, -1.779317101928757e-01, -7.336051939451567e-02, -2.025758111267412e-02, -3.448291517559102e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_otpss_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.433154773803373e+00, -2.435109660764938e+00, -1.863615931851175e+00, -1.864370610618406e+00, -4.436856006563820e-01, -4.435510344948900e-01, -2.326845361918236e-01, -1.500713179408352e-01, -9.616282794180964e-02, -7.234510894678461e-02, -2.711112573216636e-02, -2.689057646959459e-02, -5.461696861964034e-04, -2.193028551551951e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.129006208687873e-04, 3.680745913834265e-04, -1.122417999219564e-04, -1.346861098248723e-03, 1.585401755420738e-03, -1.337240723529204e-03, 5.547416083503465e-02, 4.772261375093195e-01, 5.514276481445990e-02, -1.766907510186861e+00, 1.107736517991647e+01, 5.400693449453668e+00, 1.373817352886114e+02, 4.466534975052717e+02, 2.224482897749135e+02, -5.851791890231145e-05, 6.120282239626389e-09, -1.309860480770727e-01, -4.009452927081255e-11, -1.019926189231645e-15, -1.773559742345290e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.218981761721827e-03, -5.231659226639439e-03, 1.920637081662576e-02, 1.910986104241548e-02, -5.313563984239110e-05, -5.590852776523787e-05, -4.328871023747592e-02, -5.259981438074798e-10, -1.204033918479574e-03, -7.559297200856651e-17, -1.646958090846588e-15, -1.677974643903364e-10, -3.099395700541394e-33, -6.927298824061314e-14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
