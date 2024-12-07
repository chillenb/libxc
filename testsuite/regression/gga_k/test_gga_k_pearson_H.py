
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pearson_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035145370389078e+00, 1.687198849778883e+00, 6.075202262269799e-01, 4.002501895655160e-02, 8.855167352051152e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pearson_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388871680733623e+00, -2.762114368579295e-16, 2.675685424118689e+00, 1.103584884723995e-15, 9.354723839222860e-01, 2.637731933175839e-16, 6.737865325627716e-02, 1.212555838613267e-17, 1.475861438279545e-04, 1.533623186367309e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pearson_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pearson", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.658776355450753e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.331585622649445e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.682968187740235e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.544762300436246e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.244165255462186e-06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
