
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pz_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.255706505546998e-02, -3.130080815366347e-02, -2.537071340988215e-02, -1.331501371173647e-02, -1.551534938811784e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pz_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.645523676201286e-02, -2.216561652752601e-01, -3.510438696399098e-02, -2.133454826315085e-01, -2.891492662136576e-02, -1.723847794511009e-01, -1.579046931013873e-02, -8.503670672123530e-02, -1.980255737927737e-03, -7.164136425862641e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
