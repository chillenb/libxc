
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_rc04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.184920162403562e-02, -6.762290668494074e-02, -4.196536942815710e-02, -1.279358926227337e-02, -6.104680803940322e-03, -3.168417755668601e-03, -5.398038305654028e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_rc04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.531117631297413e-02, -7.518175840331018e-02, -7.201979825223202e-02, -7.190473366711429e-02, -4.894646022746368e-02, -4.898216119461207e-02, -1.566695861693175e-02, -3.578137425861807e-01, -7.845336492337185e-03, -2.197964896938265e+00, -4.162200209100949e-03, -4.208458584240325e-03, -6.148342147311512e-05, -1.011248295608593e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
