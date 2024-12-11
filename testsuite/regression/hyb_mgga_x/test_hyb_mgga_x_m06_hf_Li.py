
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_hf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [6.028227533243162e-01, 1.807878125186238e-01, -1.075483628880488e-01, 4.824180920039964e-02, -1.682830782614241e-02, 3.642403717948241e-02, 6.223232391872034e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_hf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.654168030600219e+00, 1.649708652483341e+00, 3.714912092535358e-01, 3.732523742113040e-01, -2.218799021418471e-01, -2.113335856748564e-01, -5.354152340874854e-02, 4.548001728551482e-02, 1.267880759334848e-02, 1.479073684022999e-03, 4.901592711728606e-02, 4.740335283559552e-02, 9.879989965408172e-04, 3.893597805403889e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.890325386340024e-04, 0.000000000000000e+00, 2.882572381565853e-04, 9.273655142870310e-04, 0.000000000000000e+00, 9.248040142652861e-04, -3.796597142160303e-01, 0.000000000000000e+00, -3.724381767865106e-01, 4.265785191286263e+00, 0.000000000000000e+00, -1.335719169691490e+00, -5.027963811787846e+01, 0.000000000000000e+00, -8.568308032785307e+00, -5.707108041300243e-04, 0.000000000000000e+00, -1.267328187268999e+00, -3.910920010611421e-10, 0.000000000000000e+00, 6.243806190805820e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.211128584767054e-01, -1.201975827154897e-01, -1.910347289276526e-02, -1.930795066533286e-02, 1.438144154824676e-02, 1.252944906038419e-02, 3.353332805826083e+00, 1.488732769507488e-04, -2.940298246118314e-01, 3.052638181536251e-08, 7.411068173722407e-08, 1.606712254055283e-04, 4.153890918141848e-19, -4.367620956886795e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
