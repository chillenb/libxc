
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_fr_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.647532871554887e+01, 8.189235946127759e+00, 6.499113392259770e-01, 1.326657702882297e-01, 2.643988553994715e-02, 7.450861695526155e-03, 5.594458501715014e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_fr_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.598022222941209e+01, 2.602759952398315e+01, 1.241513123439905e+01, 1.243646096569866e+01, 7.452493719585677e-01, 7.448231005006372e-01, 2.136735868043289e-01, 4.311537226501347e-03, 3.317825907540530e-02, 4.307937776512366e-05, 4.603692231971985e-03, 4.608748798015616e-03, 2.904721861538623e-05, 1.824700315666735e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_fr_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.492819532470749e-03, 0.000000000000000e+00, 2.486873996680706e-03, 6.440087598400009e-03, 0.000000000000000e+00, 6.425516921953515e-03, 1.639475673451548e-01, 0.000000000000000e+00, 1.638206074232849e-01, 3.652139287005813e+00, 0.000000000000000e+00, 7.090935045862420e+01, 2.382596980430923e+01, 0.000000000000000e+00, 8.145807779458821e+04, 6.483870086635511e+01, 0.000000000000000e+00, 6.447058849450758e+01, 1.615360595386825e+05, 0.000000000000000e+00, 3.420886629703587e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
