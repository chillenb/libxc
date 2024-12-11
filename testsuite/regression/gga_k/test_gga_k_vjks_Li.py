
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vjks_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.640503276752238e+01, 8.103665413381250e+00, 1.225861873867316e-01, 1.317260986095184e-01, 1.971443191531770e-02, -1.840404139911429e+00, -8.141347530245141e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vjks_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.607422873813886e+01, 2.612142090329568e+01, 1.262154458136672e+01, 1.264269879343612e+01, 2.257325643252211e+00, 2.264518217780382e+00, 2.139552616743635e-01, 1.849747090906265e+00, 5.462764454243581e-02, 7.261522081362451e-01, 1.836416256994070e+00, 1.899661600357637e+00, 8.510752871960908e-01, 7.113555077710231e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vjks_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vjks", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.130939203348484e-03, 0.000000000000000e+00, 2.126328955766804e-03, 4.601245867211145e-03, 0.000000000000000e+00, 4.593322850719869e-03, -9.946068232184717e-01, 0.000000000000000e+00, -9.988644993105540e-01, 3.247054343484845e+00, 0.000000000000000e+00, -4.697788113141031e+04, -4.763792971580349e+01, 0.000000000000000e+00, -1.472626036104080e+09, -4.039979321055907e+04, 0.000000000000000e+00, -4.129436552750300e+04, -4.941685868900528e+09, 0.000000000000000e+00, -1.375415025037943e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
