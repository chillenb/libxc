
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_wb97m_v_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.551599622158522e+00, -9.912242858933651e-01, -1.829632523432268e-01, 3.236910777403232e-02, -2.431889241391604e-02, -2.048971083336413e-02, -5.093536634443863e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_wb97m_v_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.247993879192704e+00, -2.249723630420095e+00, -1.465815698854632e+00, -1.466802007567768e+00, -2.609850462833231e-01, -2.611858065240472e-01, 7.285347217718013e-02, -5.661743973971269e-02, -2.744198407359388e-02, 1.380171708665269e-01, -2.576176530189171e-02, -2.569014680220177e-02, -7.529254806494652e-04, -4.464279260239757e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.473108028549383e-04, 0.000000000000000e+00, -2.465925385711323e-04, -1.099202282997297e-03, 0.000000000000000e+00, -1.095801727147742e-03, -3.847081724510366e-02, 0.000000000000000e+00, -3.817674356730808e-02, -1.431907069343648e+00, 0.000000000000000e+00, 1.236314639843846e+01, -1.411927389317737e+01, 0.000000000000000e+00, 2.281635155117968e+03, -4.609346670131978e-05, 0.000000000000000e+00, -1.715870158418866e-02, -2.546155896317412e-11, 0.000000000000000e+00, -2.297632555002806e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.750025414663406e-02, 1.750613909923216e-02, 1.502908751564313e-02, 1.502637686764128e-02, 1.960919258925113e-03, 1.980660471157004e-03, -1.798540446528511e+00, -8.198162095510261e-06, -3.408718351856818e-02, -2.354794104528759e-08, -1.002019918973548e-08, -2.232639351362834e-05, -5.377176195444047e-20, -7.051858482490668e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
