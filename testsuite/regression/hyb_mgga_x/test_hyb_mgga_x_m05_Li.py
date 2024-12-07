
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.257992147161576e+00, -8.647937084319275e-01, -3.650995121716846e-01, -1.216852276178402e-01, -6.344974482710393e-02, -1.330621383765436e-03, -2.254192972349220e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.259247011466271e+00, -2.263013386574766e+00, -7.002133857057077e-01, -7.009147762443807e-01, -1.260103011916358e-01, -1.305211433051877e-01, -1.356443076991222e-01, -2.196876942594071e-03, -5.935171929741586e-02, -4.877316645238493e-05, -1.628378414382426e-03, -2.335561104924698e-03, -3.254178604864146e-05, -2.314013752057813e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.788101701387039e-04, 0.000000000000000e+00, -1.783304760790684e-04, -6.809740555671004e-04, 0.000000000000000e+00, -6.784094686110866e-04, -6.581975320327713e-02, 0.000000000000000e+00, -6.507259713020568e-02, -3.003862173113698e+00, 0.000000000000000e+00, -1.944863430452055e-02, -5.335606669844424e+01, 0.000000000000000e+00, -1.043702384404743e-01, -1.664586164187154e-02, 0.000000000000000e+00, -1.863902279483957e-02, -7.594037596549931e-02, 0.000000000000000e+00, -1.087131270889172e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.777664856614836e-02, 9.798018823335770e-02, -4.728645507508600e-02, -4.727247672835006e-02, -4.128306475248100e-02, -4.093886558919348e-02, -5.827191287334120e-01, 7.222183737150911e-05, -7.240764058580328e-03, 1.483347150048164e-08, 3.598912581114782e-08, 7.793732444740785e-05, 2.021691509292738e-19, 1.654917598131918e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
