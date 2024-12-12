
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_zlp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.521444740583590e-01, -5.765802667141025e-01, -3.183565166143572e-01, -1.176402421228779e-01, -4.164202087909124e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_zlp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.938868857979402e-01, -5.938868857979401e-01, -6.099327612608428e-01, -6.099327612608428e-01, -3.440795095657005e-01, -3.440795095657005e-01, 2.722074313661940e-02, 2.722074313661946e-02, 3.481530991007000e+00, 3.481530991006999e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_zlp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.340435756343013e-02, -2.680871512686027e-02, -1.340435756343013e-02, -2.075445708436837e-02, -4.150891416873675e-02, -2.075445708436837e-02, -1.778774533145077e-01, -3.557549066290154e-01, -1.778774533145077e-01, -3.509546695831536e+01, -7.019093391663073e+01, -3.509546695831536e+01, -7.135867214513841e+06, -1.427173442902768e+07, -7.135867214513841e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_zlp_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([3.996148302295787e-03, 3.949299401977191e-03, 4.460364545617617e-03, 4.387655578692052e-03, 7.649197475980726e-03, 7.621404575303804e-03, 2.874483499218374e-02, 2.874409034654511e-02, 6.110955449064808e-01, 6.110955496709125e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
