
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.643131576408926e+01, 8.173997154944660e+00, 6.414746659992570e-01, 1.323607718307269e-01, 2.665090658114368e-02, 1.243876216185658e-03, 4.422325467242587e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.591992335626407e+01, 2.596741179861679e+01, 1.232697073685816e+01, 1.234827659488298e+01, 8.307821845178975e-01, 8.307460099114986e-01, 2.135546950469496e-01, 1.886785532385685e-03, 3.396275724041523e-02, 1.900499440852579e-06, 2.086064106930153e-03, 2.055982818858707e-03, 8.479044780527390e-07, 4.285235969081894e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.470547220081814e-03, 0.000000000000000e+00, 2.464271060736274e-03, 6.770976067229292e-03, 0.000000000000000e+00, 6.755044188388462e-03, 1.156932124840879e-01, 0.000000000000000e+00, 1.153496284183991e-01, 3.464150536734745e+00, 0.000000000000000e+00, 1.558249822919174e-02, 2.287891063789596e+01, 0.000000000000000e+00, 3.160937359423632e-03, 1.665191391060904e-02, 0.000000000000000e+00, 1.543677307007911e-02, 1.536966210395184e-03, 0.000000000000000e+00, 1.564005338959624e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
