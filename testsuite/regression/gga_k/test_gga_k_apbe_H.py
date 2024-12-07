
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_apbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035474361486319e+00, 1.699317667797177e+00, 6.211616167246266e-01, 6.217320069427657e-02, 1.596956460439513e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_apbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388546203315260e+00, 7.751404892118391e-17, 2.669768589820062e+00, 3.853340727958231e-16, 8.932601486070931e-01, 6.111601692460295e-17, 8.546774759940798e-02, 2.607555770671383e-17, 2.660219437894400e-04, -1.571251329991946e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_apbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.999065449293139e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.544594751239865e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.101386884491381e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.043441935945475e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.098487983093274e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
