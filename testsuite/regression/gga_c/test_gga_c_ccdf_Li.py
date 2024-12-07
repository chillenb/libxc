
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_ccdf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.296123982974549e-02, -2.387550702758979e-02, -2.015754256245612e-02, -3.892901461966642e-02, -1.587878340523924e-02, -8.568031324045960e-03, -2.514783069001103e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_ccdf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.961855129640882e-02, -7.961855129640882e-02, -6.843732974130817e-02, -6.843732974130817e-02, -2.053009282860730e-02, -2.053009282860730e-02, -6.488016267168176e-02, -6.488016267168176e-02, -1.723346830404527e-02, -1.723346830404527e-02, -1.027739370518705e-02, -1.027739370518705e-02, -3.343166090108173e-04, -3.343166090108173e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_ccdf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.124719338366130e-05, 6.249438676732260e-05, 3.124719338366130e-05, 1.169112956773457e-04, 2.338225913546914e-04, 1.169112956773457e-04, 1.533547635678827e-17, 3.067095271357655e-17, 1.533547635678827e-17, 1.177680007168969e+01, 2.355360014337938e+01, 1.177680007168969e+01, 4.391604315825694e-11, 8.783208631651387e-11, 4.391604315825694e-11, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
