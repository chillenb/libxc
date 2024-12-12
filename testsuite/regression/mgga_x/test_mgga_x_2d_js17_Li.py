
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_js17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.817176056637241e+00, -2.231477249161224e+00, -8.034606449040180e-01, -1.111237990470901e-01, -1.239135000126741e-01, -5.923153773784422e-01, 9.717852488840970e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_js17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.799962597953962e+00, -5.808515668467038e+00, -3.102982567789539e+00, -3.107611643555224e+00, 6.631842599746993e-02, 6.684802567742069e-02, -1.305396708265111e-01, 6.055499265336556e-01, 2.368879756851104e-02, 9.789090775227370e-01, 2.308765811458806e+00, 6.071858569280004e-01, 9.251277793190697e+03, 1.208973100922231e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.041857490154147e-04, 0.000000000000000e+00, 1.049936708366101e-04, -1.198925236557998e-03, 0.000000000000000e+00, -1.192411536030763e-03, -5.635520782847684e-01, 0.000000000000000e+00, -5.635629803287692e-01, -1.582180916063197e+01, 0.000000000000000e+00, -2.971239436125520e+04, -4.195442955442442e+02, 0.000000000000000e+00, -3.799588701081567e+09, -4.701860827370946e+04, 0.000000000000000e+00, -2.551515153022634e+04, -3.070495259262504e+13, 0.000000000000000e+00, -4.473892074544984e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.153056435364765e-03, 1.151932025083771e-03, 1.682671750539816e-03, 1.681126910502795e-03, 5.580403676609097e-03, 5.576531298141249e-03, 3.084366718345668e-02, 1.707160856468683e-02, 4.280723785739769e-02, 6.964216398873832e-02, 1.687113591014092e-02, 1.667773696857365e-02, 7.376868828249847e-02, 8.779703894684937e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
