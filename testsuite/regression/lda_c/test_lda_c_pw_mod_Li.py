
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_mod_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.348344945311238e-02, -8.371482262002407e-02, -4.959806172627839e-02, -1.808617984643081e-02, -1.095911360426562e-02, -6.778651594316425e-03, -1.681738173367795e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_mod_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026427717844339e-01, -1.024808766654087e-01, -9.254539378426756e-02, -9.240668603323736e-02, -5.664537600732749e-02, -5.668890672673784e-02, -2.101623958805818e-02, -1.243112966454657e-01, -1.310473963818948e-02, -7.152742107537691e-02, -8.521702486437044e-03, -8.617314060192813e-03, -1.978391892536515e-04, -2.903012928114558e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
