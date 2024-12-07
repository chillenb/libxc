
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ernzerhof_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035145982656967e+00, 1.689056336036656e+00, 6.201116356388255e-01, 1.620334323264126e-01, 6.792402275501822e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ernzerhof_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388869435693111e+00, 6.659772949697131e-17, 2.667485732798653e+00, 8.380914736175284e-16, 8.579777958265437e-01, 1.859057542671762e-16, -2.052543641349857e-01, 1.148206512395927e-17, -6.944458112664605e-01, 3.710693894393365e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ernzerhof_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.663785187073800e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.856250294911614e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.833711886390706e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.350088102068172e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.459526767016237e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
