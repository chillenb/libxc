
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.175302394191925e-02, -6.503247594912004e-02, -1.344636918403049e-01, -8.872311886379173e-04, -2.644561266435018e-02, 1.910519372512543e-02, 3.225932487966175e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.857971664614295e-02, -7.827263241422362e-02, -6.773543845701885e-02, -6.744652582822691e-02, 3.723272567891679e-02, 3.660502088244253e-02, 3.029439314496633e-03, 6.914288548868508e-01, 3.753300211000526e-03, 4.174968764435884e-01, 3.304218301656967e-02, 3.369077493626815e-02, 5.471545540246952e-04, 9.377303008948020e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.376736742575625e-05, 0.000000000000000e+00, -2.418028723287283e-05, -9.661557153945397e-05, 0.000000000000000e+00, -9.610369573276538e-05, 2.876988979356853e-01, 0.000000000000000e+00, 2.868422682467791e-01, -1.306340831946030e+00, 0.000000000000000e+00, -7.969150854260940e+01, 1.755658116857375e+02, 0.000000000000000e+00, -8.753106757684397e+05, -4.060114406220772e+00, 0.000000000000000e+00, -1.948628847538483e+02, -1.381107520966771e+01, 0.000000000000000e+00, -6.228088076183277e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.233018426880456e-05, -3.132927026407638e-05, -6.069212618413016e-04, -6.176572518614914e-04, -3.483010170066237e-02, -3.531534023132490e-02, -1.219546844198712e-01, 2.753647212827356e-03, -2.962076206041587e-01, 3.631734391555109e-04, 6.033898811395014e-05, 2.831300034566004e-03, 1.676886258295104e-09, 4.552792679096311e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
