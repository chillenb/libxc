
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_gombas_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_gombas", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.364001691474190e-02, -8.394054152667015e-02, -5.177437483592197e-02, -3.731285804203417e-02, -2.400739896073713e-02, -8.815790564371201e-03, -1.937527213945775e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_gombas_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_gombas", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.027116695555863e-01, -1.027116695555863e-01, -9.257476847604357e-02, -9.257476847604357e-02, -5.809309511452845e-02, -5.809309511452845e-02, -4.252212065374269e-02, -4.252212065374269e-02, -2.838844138573876e-02, -2.838844138573876e-02, -1.119556581060186e-02, -1.119556581060186e-02, -2.580540787793108e-04, -2.580540787793108e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
