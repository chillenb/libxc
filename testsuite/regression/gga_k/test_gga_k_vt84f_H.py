
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vt84f_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035329722810326e+00, 1.978098644001001e+00, 1.042156888286706e+00, 5.136667188776343e-01, 6.849985247851196e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vt84f_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388200523931796e+00, 3.746728006936734e-16, 1.959918422049748e+00, 5.777552149374144e-16, 4.769911952897262e-01, 2.243530036776132e-16, -4.858908984595555e-01, 2.423551646905526e-16, -6.849984310299648e-01, -1.359825401449263e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vt84f_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.159531112721208e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.209824526395409e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.751618621221015e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.510556935629850e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646359429231e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
