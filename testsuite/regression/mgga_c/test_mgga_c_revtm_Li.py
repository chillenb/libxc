
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.346251621985899e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808613505659574e-02, -1.095911360424506e-02, -1.550193581521006e-11, -8.300292557722951e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026705243451290e-01, -1.025086418150166e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101629755018789e-02, -1.243109553808686e-01, -1.310473963821593e-02, -7.152742107294510e-02, -1.113606823237239e-10, -8.809083555528910e-11, -1.073246087570411e-18, -5.118744759909199e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.359498074226073e-04, 2.718996813693698e-04, 1.359498073632867e-04, 5.803800301148971e-04, 1.160760060229794e-03, 5.803800301148971e-04, 1.624082993170471e-01, 3.248165986340941e-01, 1.624082993170471e-01, 3.449442953256744e+00, 6.898885908559577e+00, 3.449443026097574e+00, 1.223550734328922e+02, 2.447101468657844e+02, 1.223550734328922e+02, 1.306031819869226e-08, 2.843160858755609e-08, 1.421580494193043e-08, 3.696258305292843e-15, 2.024865339056517e-15, 1.479558879703884e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.665101733471457e-09, -2.665101733471456e-09, -9.588257702779593e-77, -9.588257702779591e-77, -3.056880835874814e-69, -3.056880835874812e-69, -3.821009485070113e-10, -3.821009485069267e-10, -2.940021288672253e-25, -2.940021286296991e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
