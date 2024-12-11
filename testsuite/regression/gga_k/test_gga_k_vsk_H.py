
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vsk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035145150870842e+00, 1.687056126450167e+00, 6.147200977371740e-01, 2.885696347759131e-01, 6.850969807201015e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vsk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388872482464911e+00, 8.634822798014166e-17, 2.674355267728048e+00, 7.235991703733957e-16, 8.729729406101847e-01, 2.481046080988137e-16, -7.987598754839226e-01, 8.749105503275121e-18, -6.848398159609475e-01, -9.536541453089251e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vsk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.656985416813228e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.382321121786994e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.309981872259652e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.440440155377357e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459650832795249e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
