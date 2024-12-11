
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_scan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.481295426735947e+00, -9.779013085367365e-01, -1.757124211365625e-01, -1.357290771342137e-01, -3.935195389404488e-02, -3.612441797238099e-03, -2.639775894306764e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_scan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.069753381291439e+00, -2.071672100151825e+00, -1.505824004817491e+00, -1.506656812413736e+00, -2.440542701816893e-01, -2.444274896672222e-01, -1.855567901164771e-01, 1.415776247620468e+00, -5.824838738610781e-02, 5.087171915266292e+00, 3.084567006857779e+01, 1.406240207651677e+00, 5.045370851219140e+04, -1.284457988862074e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_scan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.883639775374828e-04, 0.000000000000000e+00, -2.872552267618423e-04, -1.990432518871537e-03, 0.000000000000000e+00, -1.979719213455282e-03, -3.248755763737349e-01, 0.000000000000000e+00, -3.252467026473316e-01, -3.570965025794707e+00, 0.000000000000000e+00, -3.645612235993509e+04, -1.501259968469230e+02, 0.000000000000000e+00, -1.031708203690244e+10, -1.400895366028628e+04, 0.000000000000000e+00, -3.101941507671846e+04, -2.319738199125620e+09, 0.000000000000000e+00, -3.308938092075643e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_scan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.362948413197895e-02, 1.361433629183823e-02, 2.963485829526353e-02, 2.955882278625317e-02, 1.783633349998115e-03, 1.881115283426766e-03, 1.287778329918326e-01, 4.658753259031679e-01, 4.838827949584323e-02, 4.203552731743675e+00, 2.080528103532292e-01, 4.509706474135126e-01, 2.816534592406493e-01, 2.129936396140093e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
