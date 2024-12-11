
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_yt65_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.036056551495886e+00, 1.729204432396152e+00, 6.592623569014344e-01, 1.415889948219185e-01, 1.370882528233145e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_yt65_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.387960499374950e+00, 2.441745085645808e-16, 2.631566787359552e+00, 1.588370579438539e-16, 8.325720898701052e-01, 7.173708413020755e-17, -3.519188098244710e-02, 7.181674685552727e-17, -1.368521166025185e-01, -2.729643464770859e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_yt65_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.385798367223624e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.163271346551712e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.813598546549023e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.052328093716774e+01, 0.000000000000000e+00, 0.000000000000000e+00, 2.919292800167867e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
