
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_b88_v2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88_v2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.740530981999640e+00, -1.228932393848802e+00, -3.705843220417818e-01, -1.060475146040219e-01, -3.013152255329995e-02, -1.675600088835746e-02, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_b88_v2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88_v2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.180868599222957e+00, -2.182988792593075e+00, -1.463004127274326e+00, -1.464364394102451e+00, -2.783116732252867e-01, -2.781792027341013e-01, -1.504651807549181e-01, -2.827849512055973e-02, -2.627634400382427e-02, -2.423873708293871e-17, -3.180264017807980e-02, -3.127385312468055e-02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_b88_v2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88_v2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.730928182603701e-04, 0.000000000000000e+00, -2.721887435666053e-04, -1.029562825946136e-03, 0.000000000000000e+00, -1.026338536975849e-03, -1.144370886574953e-01, 0.000000000000000e+00, -1.144070912408524e-01, -3.052283043966123e+00, 0.000000000000000e+00, -1.387209132402873e-12, -5.541138683465997e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.220962505848687e-11, 0.000000000000000e+00, -6.211070618559189e-12, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
