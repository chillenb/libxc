
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.348387038075551e-02, -8.371516842999983e-02, -4.959817977182925e-02, -1.808611992110955e-02, -1.095909922859326e-02, -6.778651964598665e-03, -1.681738177383024e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026432641442106e-01, -1.024813685297047e-01, -9.254580573786172e-02, -9.240709758920679e-02, -5.664553279612767e-02, -5.668906359864599e-02, -2.101615291928950e-02, -1.243125303698876e-01, -1.310471672624482e-02, -7.152770966862954e-02, -8.521703160619442e-03, -8.617314737328054e-03, -1.978391906071125e-04, -2.903012910727799e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
