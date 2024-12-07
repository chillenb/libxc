
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.798814235449836e+00, -1.289905741930406e+00, -4.336666063285744e-01, -1.602559205577097e-01, -8.262002492592530e-02, -2.239977711651121e-02, -4.185836410040993e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.236465210089917e+00, -2.238605147038414e+00, -1.510087272629675e+00, -1.511459652224542e+00, -3.917030611946589e-01, -3.919157234728979e-01, -2.049939691520572e-01, -2.846694429857098e-02, -7.420026552450500e-02, -9.046943941624333e-04, -2.992786489447230e-02, -2.971344375013205e-02, -6.042858435317953e-04, -4.295923632977178e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.757967565726553e-04, 0.000000000000000e+00, -2.748423906172459e-04, -1.101689820284482e-03, 0.000000000000000e+00, -1.098133770367306e-03, -9.044664215238385e-02, 0.000000000000000e+00, -9.024341944287204e-02, -4.253794906736579e+00, 0.000000000000000e+00, -3.751293459832124e-01, -7.868838471753736e+01, 0.000000000000000e+00, -2.400299842239697e+00, -3.811979420777318e-01, 0.000000000000000e+00, -3.559774340680800e-01, -1.747330634264166e+00, 0.000000000000000e+00, -2.501124324713428e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
