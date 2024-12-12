
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_tpsslyp1w_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.127008271485483e-01, -6.402119459890344e-01, -3.771624731075351e-01, -1.349020783051429e-01, -7.802302529290104e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_tpsslyp1w_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.388240621642412e-01, -2.450141668596068e-01, -7.840856234589246e-01, -2.519781966670389e-01, -4.708402625998552e-01, -1.978538093371852e-01, -1.294574185938628e-01, -5.076325836282430e-02, -1.037361273669463e-02, -3.624513323217621e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.483026404810098e+00, 2.164366886827692e-02, 1.622653450346063e-02, -8.404106579035130e-02, 3.438719061436823e-02, 2.575692157951031e-02, -1.448441320550674e-01, 2.950210920792956e-01, 2.212592819169965e-01, -5.647887788951197e+00, 1.260062407992866e+01, 9.450452070551076e+00, -5.540657391391956e+00, 5.296779305705816e-18, 3.972578537740708e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.068800958037017e+01, 0.000000000000000e+00, 1.444921603544164e-01, 0.000000000000000e+00, 4.272293233047415e-02, 0.000000000000000e+00, 3.345156569908141e-04, 0.000000000000000e+00, 4.555983676671474e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
