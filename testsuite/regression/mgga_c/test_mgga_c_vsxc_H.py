
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_vsxc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.316399065773768e-16, -3.993758772738646e-02, -9.405150490646618e-03, -6.279997796387651e-01, -2.525929666525467e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_vsxc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.927386162941775e-02, -1.780516748108547e-01, -8.226131989342096e-02, -1.700134065408582e-01, -3.238752727176639e-02, -1.324526142118897e-01, 3.306186185756145e-01, 8.870957714723926e-04, -7.396492808211014e-03, 5.006619124143103e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.833522096269956e+00, 0.000000000000000e+00, 1.245866268931503e+19, 5.396118181417963e-02, 0.000000000000000e+00, 1.190340920948666e+19, 1.951670032053256e-01, 0.000000000000000e+00, 9.277144152393052e+18, 1.380508655721946e+03, 0.000000000000000e+00, -3.359480900263832e+15, 7.772630614360071e+04, 0.000000000000000e+00, -1.006348447624119e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.757906575347535e+00, 6.645380975032842e+03, 4.768440275702799e-02, 6.349230034551919e+03, 2.603957204700307e-02, 4.948391402179763e+03, -1.267692177115803e+00, 3.697983555439378e+00, 3.657504555554873e-03, -6.800912571632355e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
