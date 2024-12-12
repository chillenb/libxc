
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_l_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.926635104512645e-01, -5.433069100624413e-02, 1.867934001111181e-04, -1.706014655438134e-03, -1.111077935203375e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_l_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.673610180309439e-01, -3.845499644966799e+02, -1.133657596881442e-01, -4.751682256956246e+02, -7.780111366441404e-02, 6.804861870992502e+02, 8.088656389561899e-02, 3.896928759581836e+01, -1.412004064009900e-02, -5.000085489850875e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.370050642353931e-01, -6.740101284707860e-01, -3.370050642353931e-01, -6.527450080693597e-03, -1.305490016138719e-02, -6.527450080693597e-03, 6.890709096233674e-02, 1.378141819246735e-01, 6.890709096233674e-02, 1.001020024982358e+00, 2.002040049964726e+00, 1.001020024982358e+00, 8.770165022829691e-03, 1.754033002894922e-02, 8.770165022829691e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.742866205778410e+00, 2.741484005125279e+00, 8.364050175010097e-02, 8.308003991506735e-02, 5.762291051377533e-02, 5.750711478869824e-02, -1.079524782467918e-01, -1.079518316142585e-01, -4.159813526589700e-05, -4.159813550458220e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
