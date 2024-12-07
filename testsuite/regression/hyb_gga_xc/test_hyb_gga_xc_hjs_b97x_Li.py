
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_b97x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.397658555649017e+00, -9.913493627634820e-01, -3.224155746689635e-01, -1.473385672539712e-01, -6.995582662167887e-02, -2.874425652801429e-02, -1.790559887859811e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_b97x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.839319698207365e+00, -1.840784204056661e+00, -1.282423626942788e+00, -1.283368779298056e+00, -2.477112806363727e-01, -2.476260777527932e-01, -1.936046551767585e-01, -1.338833800402825e-01, -6.735613125460094e-02, 3.428185832405125e-01, -3.799289815613630e-02, -3.774294289854151e-02, -1.135871182268017e-15, -1.343631423570552e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_b97x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.724078199310725e-05, 9.190971700708733e-05, -4.690255146134326e-05, -2.581514033802461e-04, 2.980993506782570e-04, -2.567566079035999e-04, -8.848684579546928e-02, 6.249948659585063e-03, -8.846648279537793e-02, 1.854502510012801e+00, 6.762268918356340e+00, 1.918211700570138e+00, -5.040395061904885e+01, 2.258698854598489e+01, 1.129349427299244e+01, -1.477549182801953e+00, 3.357174600576258e-04, -1.381246833095986e+00, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
