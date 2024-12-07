
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbeint_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.215304078570529e-02, -2.398810683065708e-02, -1.833958438102816e-02, -1.267532795312722e-02, -1.575746294245629e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbeint_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.686211036098544e-02, 4.797489174851297e-02, -2.909800003672524e-02, 8.555984636966558e+00, -1.124902814730381e-02, 3.846768303317852e+00, -8.931157145414774e-03, -5.848643567665016e-02, -2.004338848365761e-03, -6.542002833382654e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbeint_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.284601083725443e-02, 2.569202167450885e-02, 1.284601083725443e-02, 3.998425169551916e-04, 7.996850339103832e-04, 3.998425169551916e-04, -2.812068554597679e-02, -5.624137109195357e-02, -2.812068554597679e-02, -8.219854969927901e-01, -1.643970993985580e+00, -8.219854969927901e-01, -7.910159637894207e-39, -1.582031927578841e-38, -7.910159637894207e-39]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
