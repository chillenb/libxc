
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037422781146677e+00, -1.413111899517623e+00, -3.142841180935740e-01, -1.841757860718493e-01, -7.184056513608880e-02, -5.830961010782742e-03, -1.821095506134382e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r2scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.719573523920549e+00, -2.722014612766085e+00, -1.892035111043257e+00, -1.893636405881253e+00, -1.222361056734063e-01, -1.526914594436302e-01, -2.457883112651349e-01, -8.946716524149101e-03, -9.908290499170902e-02, -9.360905506931836e-05, -1.102742463027921e-02, -9.419140019194989e-03, -5.937321680114718e-05, -4.646340229538664e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.260008312118228e-06, 0.000000000000000e+00, 7.224577549782167e-06, 4.578689914843173e-05, 0.000000000000000e+00, 4.561034488914327e-05, -3.737429830401076e-01, 0.000000000000000e+00, -3.347345741522950e-01, 6.865366266803022e-02, 0.000000000000000e+00, 2.314466608116319e+01, 7.211252344146685e+00, 0.000000000000000e+00, 4.529235417260388e+03, 2.349291921896267e+01, 0.000000000000000e+00, 2.069010770754756e+01, 4.131024424486666e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.757568872116156e-02, 1.579245594916505e-02, 0.000000000000000e+00, -2.024183146400337e-06, 0.000000000000000e+00, 1.952063728273576e-06, -3.535062557603489e-07, -1.861327074876376e-06, -3.847404375214329e-19, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
