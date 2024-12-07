
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_revapbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035474828394857e+00, 1.700441552421118e+00, 6.243152120830032e-01, 6.943864747207856e-02, 1.986748728242187e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_revapbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388544493156941e+00, 5.965219544462055e-16, 2.665889540878474e+00, 6.109657989454914e-18, 8.836343324414553e-01, 5.624158662470719e-17, 8.380217837332035e-02, 2.335788503152125e-17, 3.307954193091943e-04, -1.885323808587771e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_revapbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.002882237470176e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.811778178617891e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.426395696496178e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.593915582762362e+00, 0.000000000000000e+00, 0.000000000000000e+00, 2.631944725082649e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
