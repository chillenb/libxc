
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zpbesol_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.218947046853084e-02, -2.538073245806225e-02, -2.152742111652106e-02, -1.327146400798257e-02, -1.569853905510373e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zpbesol_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.681497183693173e-02, 1.026762959083052e+00, -2.754677320642913e-02, 1.268613280093850e+02, -1.199744866300231e-02, 9.544084592813986e+01, -1.569772379917025e-02, -6.727548664161004e-02, -2.001690237919110e-03, -6.965489951221249e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zpbesol_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.138220547175045e-02, 2.276441094350090e-02, 1.138220547175045e-02, -9.020537286017007e-04, -1.804107457203402e-03, -9.020537286017007e-04, -3.229625429314863e-02, -6.459250858629728e-02, -3.229625429314863e-02, -2.274319303876692e-03, -4.548638607753384e-03, -2.274319303876692e-03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
