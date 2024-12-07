
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.382820466517786e+00, -1.207221334533612e+00, -4.837436607961116e-01, -1.469842897212987e-01, -8.663174149809015e-02, -2.628530496217920e-01, -3.621820110030730e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.870106372025604e+00, -1.871879217698441e+00, -1.753300059340809e+00, -1.754909164896340e+00, -4.215636884246048e-01, -4.210715872665167e-01, -2.240597308598702e-01, -8.387133750281309e-02, -8.458880352353103e-02, -1.133996546957211e-02, -2.205238617148054e-01, -8.373587637512539e-02, -2.771948095652257e+00, -6.093688437276258e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.182030332242547e-07, 0.000000000000000e+00, -6.181868670478816e-07, -9.884173150029635e-04, 0.000000000000000e+00, -9.856549719301478e-04, -8.039547345980032e-02, 0.000000000000000e+00, -8.056932025508019e-02, -2.063919138750749e+00, 0.000000000000000e+00, -7.588227641724953e+02, -6.379009102295439e+01, 0.000000000000000e+00, -3.696591584234281e+07, -2.282280154807924e+02, 0.000000000000000e+00, -6.745578539529531e+02, -4.163062190540896e+05, 0.000000000000000e+00, -3.527916860990373e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.002434242548513e-05, -1.005120060744718e-05, -5.364051350233188e-03, -5.362730434381820e-03, -6.048910462736676e-03, -6.054259907176016e-03, -2.475879790791295e-02, -3.028580083060356e-03, -4.767264989864048e-02, -4.706632263698391e-03, -1.059204310221995e-03, -3.062805265680433e-03, -1.579570578581893e-05, -4.809344084434759e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.207789576155275e-05, 3.216384194383160e-05, 1.716496432074618e-02, 1.716073739002180e-02, 1.935651348075736e-02, 1.937363170296324e-02, 7.922815330532153e-02, 9.691456265793137e-03, 1.525524796756495e-01, 1.506122324383486e-02, 3.389453792710385e-03, 9.800976850177385e-03, 5.054625851462057e-05, 1.538990107019123e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
