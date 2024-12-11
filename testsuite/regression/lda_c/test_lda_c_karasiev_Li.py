
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_karasiev_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.222331699912616e-02, -8.234682707543385e-02, -4.866874081036634e-02, -1.848066382328936e-02, -1.147153504730807e-02, -6.845832028754054e-03, -1.386820396232257e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_karasiev_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_karasiev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.015289804724482e-01, -1.013262310844674e-01, -9.126020253864046e-02, -9.109014251608336e-02, -5.544715280939853e-02, -5.549574966411715e-02, -2.129552557399012e-02, -1.095749239990211e-01, -1.365885348431474e-02, -6.737716159207689e-02, -8.685275930882719e-03, -8.783414862065764e-03, -1.615698809823128e-04, -2.490744573318919e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
