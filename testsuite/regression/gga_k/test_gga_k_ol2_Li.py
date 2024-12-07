
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ol2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.644597751469509e+01, 8.158430402027051e+00, 8.014467634847147e-01, 1.328480732067290e-01, 2.849221729779047e-02, 3.434079395528307e-01, 1.507666191675021e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ol2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.612274338687933e+01, 2.617066859695102e+01, 1.236690636301037e+01, 1.238849593250604e+01, 3.836870933292474e-01, 3.819925561672373e-01, 2.152118282265283e-01, -3.386283238966640e-01, 2.624913108271132e-02, -1.344686595924767e-01, -3.357491143720067e-01, -3.475221579557333e-01, -1.576047627503846e-01, -1.317316059260185e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ol2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.164656190584546e-03, 0.000000000000000e+00, 2.158875288781851e-03, 6.423886681825652e-03, 0.000000000000000e+00, 6.407554810473262e-03, 4.615306182294883e-01, 0.000000000000000e+00, 4.621197149687729e-01, 2.961625009060421e+00, 0.000000000000000e+00, 8.699790681070952e+03, 4.647339907762770e+01, 0.000000000000000e+00, 2.727085252420157e+08, 7.481638663529130e+03, 0.000000000000000e+00, 7.647286004088990e+03, 9.151270127776741e+08, 0.000000000000000e+00, 2.547064861200060e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
