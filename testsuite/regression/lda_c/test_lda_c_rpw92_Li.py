
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_rpw92_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.348344945311238e-02, -8.371482262002407e-02, -4.959806172627839e-02, -1.810394433964872e-02, -1.091216740925674e-02, -6.778651594316439e-03, -1.682478994025958e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_rpw92_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026427717844339e-01, -1.024808766654087e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732749e-02, -5.668890672673784e-02, -2.107594816780232e-02, -1.241217304718088e-01, -1.306010929028550e-02, -7.216009645200550e-02, -8.521702486444398e-03, -8.617314060185283e-03, -1.984351997157318e-04, -2.889964712186931e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
