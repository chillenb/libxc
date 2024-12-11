
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ol1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.642713869251945e+01, 8.166571576288963e+00, 8.115612270933856e-01, 1.325407068063027e-01, 2.877092391495718e-02, 3.439215006094232e-01, 1.507730730264144e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ol1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.598185387333169e+01, 2.602952266459156e+01, 1.230105293470095e+01, 1.232252405089874e+01, 3.826852632215777e-01, 3.809998097128194e-01, 2.140683392006728e-01, -3.384719928655450e-01, 2.614340445250297e-02, -1.344653883702626e-01, -3.355858606835486e-01, -3.473570726737177e-01, -1.576023950218727e-01, -1.317300665226420e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ol1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.352436606140533e-03, 0.000000000000000e+00, 2.346281237129176e-03, 6.842400391581170e-03, 0.000000000000000e+00, 6.825218412078838e-03, 4.701866093346068e-01, 0.000000000000000e+00, 4.707703224254132e-01, 3.268546492841681e+00, 0.000000000000000e+00, 8.706104067134545e+03, 4.772121200624385e+01, 0.000000000000000e+00, 2.727185009635903e+08, 7.487371432341580e+03, 0.000000000000000e+00, 7.653004171556669e+03, 9.151476662472548e+08, 0.000000000000000e+00, 2.547109560978589e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
