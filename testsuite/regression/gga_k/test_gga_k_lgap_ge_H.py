
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lgap_ge_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.036620030788646e+00, 1.699447008104956e+00, 6.260269972625329e-01, 1.199690448534504e-01, 8.099257263003803e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lgap_ge_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.389338826754550e+00, -9.536047380478885e-17, 2.669094806404696e+00, 2.172761779280240e-16, 8.653025486671143e-01, 1.303396004674167e-17, -4.111347258901073e-02, 6.506139608430476e-17, -1.787728544565358e+00, -2.531966765997275e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lgap_ge_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.712130110218932e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.585904190058330e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.889056764314415e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.713391407509042e+01, 0.000000000000000e+00, 0.000000000000000e+00, 2.507191667388449e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
