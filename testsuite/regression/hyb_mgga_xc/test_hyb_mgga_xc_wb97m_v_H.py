
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_wb97m_v_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.797559324753309e-01, -3.068362786857489e-01, -1.666970971974986e-01, -3.180610388791186e-02, -2.494620288643629e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_wb97m_v_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.688322028380880e-01, -1.026968377325026e+00, -3.847477101755215e-01, -4.221757496454090e-01, -1.900817498725875e-01, -1.643402459892462e-01, -4.719631944297666e-02, -1.144295968997714e-01, -3.183086418395492e-03, -3.488759647776462e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.486421389356527e-02, 0.000000000000000e+00, 1.318224990733493e+18, -2.123095256285816e-02, 0.000000000000000e+00, 2.436137661415425e+18, -1.432481902675864e-01, 0.000000000000000e+00, 1.395609493334604e+18, -6.565260099987642e+00, 0.000000000000000e+00, 8.529528593277432e+17, -3.609486077743213e+03, 0.000000000000000e+00, 4.766371941042716e+16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.372853481056422e-01, -8.645936451804182e+00, -1.131088180401155e-01, 9.433969713156419e+04, -7.521447536688323e-02, -3.039917234184657e+03, 5.235388148795549e-03, -3.183379874025356e+05, 1.328917830043100e-06, -5.087445963465267e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
