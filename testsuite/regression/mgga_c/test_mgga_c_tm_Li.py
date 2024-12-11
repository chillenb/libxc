
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.346209774030460e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808612568817779e-02, -1.095911360423736e-02, -5.892600477120597e-12, -8.300292557722951e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026712147700619e-01, -1.025093324919725e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101631195730691e-02, -1.243108862723517e-01, -1.310473963822716e-02, -7.152742107203552e-02, -4.266538431791472e-11, -3.375403351533873e-11, -1.026864148341892e-18, -7.910146580446758e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.386659375585529e-04, 2.773319432983116e-04, 1.386659374939065e-04, 5.969146803740697e-04, 1.193829360748139e-03, 5.969146803740697e-04, 1.796789276683559e-01, 3.593578553367118e-01, 1.796789276683559e-01, 4.170872753113913e+00, 8.341745508711966e+00, 4.170872826157728e+00, 1.681681181378362e+02, 3.363362362756724e+02, 1.681681181378362e+02, 4.971943069725646e-09, 1.079340474356706e-08, 5.396702538475225e-09, 1.574454197631843e-15, -1.923003370021578e-15, -7.472775817567132e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.665040917310507e-09, -2.665040917310506e-09, -2.065727501183380e-87, -2.065727501183380e-87, -6.585850116102372e-80, -6.585850116102367e-80, -3.821010084090763e-10, -3.821010084089918e-10, -2.940021288673111e-25, -2.940021286297848e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
