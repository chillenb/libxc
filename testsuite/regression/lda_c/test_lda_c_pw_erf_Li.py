
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_erf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.641321475876788e-02, -7.464382178672886e-02, -2.932284451293403e-02, -2.181707326552904e-03, -5.863111304408447e-05, -1.155254222102650e-05, -5.041114929465150e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_erf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.721168210475505e-02, -9.705721322710813e-02, -8.549071037006632e-02, -8.536196514618076e-02, -3.906032907885920e-02, -3.909178928534883e-02, -4.437673109572396e-03, -3.120624469963885e-02, -1.461731617439406e-04, -3.373745927468258e-03, -2.286674053157365e-05, -2.337052112703324e-05, -6.852329975274652e-11, -1.907199189226630e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
